import requests
import copy
from io import BytesIO
from PIL import Image
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, DPMSolverMultistepScheduler
from torchvision import transforms as tfms
from tqdm.auto import tqdm

# Set up - set seed to make results reproducible
model_id = "stabilityai/stable-diffusion-2-1-base"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


# Hyperparameters
strength = 0.5              # strength balance between real image and text prompt
guidance_scale = 7.5           # scale for classifier free guidance
prompt = ["An oil painting of a man on a bench"]

def download_image(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")
img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
init_image = download_image(img_url).resize((512, 512))


# Initialize pipelines
vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")


vae.to(device)
text_encoder.to(device)
unet.to(device)

def get_timesteps(scheduler, strength=0.5):
    """
    Get new scheduler with timesteps from [strength, 0]
    Args:
    - scheduler
    - strength: strength for SEdit
    """
    scheduler.set_timesteps(num_inference_steps=scheduler.config.num_train_timesteps)
    new_scheduler = copy.deepcopy(scheduler)
    assert 0.0 <= strength <= 1.0, "strength must be in [0.0, 1.0]"
    new_timesteps_max = int(strength * new_scheduler.config.num_train_timesteps)
    new_timesteps = new_scheduler.timesteps[-new_timesteps_max:]
    new_scheduler.set_timesteps(timesteps=new_timesteps)
    return new_scheduler 

# VAE util
def pil_to_latent(input_im):
    # Single image -> single latent in a batch (so size 1, 4, 64, 64)
    with torch.no_grad():
        latent = vae.encode(tfms.ToTensor()(input_im).unsqueeze(0).to(device)*2-1) # Note scaling
    return 0.18215 * latent.latent_dist.sample()

def latents_to_pil(latents):
    # bath of latents -> list of images
    latents = (1 / 0.18215) * latents
    with torch.no_grad():
        image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images

# text_cond and uncond embeddings
text_input = tokenizer(prompt, max_length=tokenizer.model_max_length, truncation=True, padding="max_length", return_tensors="pt")
text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
uncond_input = tokenizer("", max_length=tokenizer.model_max_length, padding="max_length", return_tensors="pt")
uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
text_embeddings = torch.cat([uncond_embeddings, text_embeddings], dim=0)


# pipeline
# 1. get init latent of init image
# 2. get new scheduler with strength to solve SDE
# 3. add noise to init latent to the maximum noise level of current scheduler
# 3. denoise step with text guidance 



# prepare latent and add noise
init_latent = pil_to_latent(init_image)

new_scheduler = get_timesteps(scheduler, strength)
noise = torch.randn_like(init_latent)
max_noise_level = new_scheduler.timesteps[0]
latents = new_scheduler.add_noise(init_latent, noise, torch.tensor([max_noise_level]))

for t in tqdm(new_scheduler.timesteps):
    latent_model_input = torch.cat([latents]*2, dim=0)

    with torch.no_grad():
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

    # perform guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    # compute the previous noisy sample x_t -> x_t-1
    latents = new_scheduler.step(noise_pred, t, latents).prev_sample


# visualize image
latents_to_pil(latents)[0]

    