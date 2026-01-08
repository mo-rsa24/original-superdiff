import matplotlib.pyplot as plt
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel

device = torch.device("cuda")
dtype  = torch.float16

vae = AutoencoderKL.from_pretrained(
    "CompVis/stable-diffusion-v1-4", subfolder="vae",
    torch_dtype=dtype, use_safetensors=True
).to(device)

tokenizer = CLIPTokenizer.from_pretrained(
    "CompVis/stable-diffusion-v1-4", subfolder="tokenizer"
)

text_encoder = CLIPTextModel.from_pretrained(
    "CompVis/stable-diffusion-v1-4", subfolder="text_encoder",
    torch_dtype=dtype, use_safetensors=True
).to(device)

unet = UNet2DConditionModel.from_pretrained(
    "CompVis/stable-diffusion-v1-4", subfolder="unet",
    torch_dtype=dtype, use_safetensors=True
).to(device)

from PIL import Image
from diffusers import EulerDiscreteScheduler

scheduler = EulerDiscreteScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")

@torch.no_grad
def get_image(latents, nrow, ncol):
    image = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
    image = (image / 2 + 0.5).clamp(0, 1).squeeze()
    image = (image.permute(0, 2, 3, 1) * 255).to(torch.uint8)
    rows = []
    for row_i in range(nrow):
        row = []
        for col_i in range(ncol):
            i = row_i*nrow + col_i
            row.append(image[i])
        rows.append(torch.hstack(row))
    image = torch.vstack(rows)
    return Image.fromarray(image.cpu().numpy())

@torch.no_grad
def get_text_embedding(prompt):
    text_input = tokenizer(
        prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
    )
    return text_encoder(text_input.input_ids.to(device))[0]

from torch.nn.attention import SDPBackend, sdpa_kernel
from diffusers.models.attention_processor import AttnProcessor

@torch.no_grad
def get_vel(t, sigma, latents, embeddings, eps=None, get_div=False):
    # unet.set_attn_processor(AttnProcessor())
    t = t.to(device, dtype=torch.float16)

    def v(_x, _e):
        _x = _x.to(device=device, dtype=dtype)
        _e = _e.to(device=device, dtype=dtype)

        denom = torch.sqrt(sigma * sigma + 1.0)  # stays fp16 now
        x_in = _x / denom

        with torch.autocast("cuda", dtype=dtype):
            return unet(x_in, t, encoder_hidden_states=_e).sample
    # v = lambda _x, _e: unet(_x / ((sigma**2 + 1) ** 0.5), t, encoder_hidden_states=_e).sample
    embeds = torch.cat(embeddings)
    latent_input = latents
    if get_div:
        with torch.enable_grad():
            with sdpa_kernel(SDPBackend.MATH):
                vel, div = torch.func.jvp(v, (latent_input, embeds), (eps, torch.zeros_like(embeds)))
                div = -(eps*div).sum((1,2,3))
    else:
        with torch.no_grad():
            vel = v(latent_input, embeds)
            div = None

    return vel, div
obj_prompt = ["A Dog On The Left"]
bg_prompt = ["A Cat On The Right"]

height = 512  # default height of Stable Diffusion
width = 512  # default width of Stable Diffusion
num_inference_steps = 100  # Number of denoising steps MUST BE > 1000
guidance_scale = 7.5  # Scale for classifier-free guidance
generator = torch.cuda.manual_seed(1)  # Seed generator to create the initial latent noise
# batch_size = len(obj_prompt)
batch_size = 6

obj_embeddings = get_text_embedding(obj_prompt * batch_size)
bg_embeddings = get_text_embedding(bg_prompt * batch_size)
uncond_embeddings = get_text_embedding([""] * batch_size)
latents = torch.randn(
    (batch_size, unet.config.in_channels, height // 8, width // 8),
    generator=generator,
    device=device,
    dtype=dtype
)
scheduler.set_timesteps(num_inference_steps)
latents = latents * scheduler.init_noise_sigma

lift = 0.0
ll_obj = torch.ones((num_inference_steps + 1, batch_size), device=device)
ll_bg = torch.ones((num_inference_steps + 1, batch_size), device=device)
kappa = 0.5 * torch.ones((num_inference_steps + 1, batch_size), device=device)
for i, t in enumerate(scheduler.timesteps):
    dsigma = scheduler.sigmas[i + 1] - scheduler.sigmas[i]
    sigma = scheduler.sigmas[i]
    vel_obj, _ = get_vel(t, sigma, latents, [obj_embeddings])
    vel_bg, _ = get_vel(t, sigma, latents, [bg_embeddings])
    vel_uncond, _ = get_vel(t, sigma, latents, [uncond_embeddings])

    noise = torch.sqrt(2 * torch.abs(dsigma) * sigma) * torch.randn_like(latents)
    dx_ind = 2 * dsigma * (vel_uncond + guidance_scale * (vel_bg - vel_uncond)) + noise
    kappa[i + 1] = (torch.abs(dsigma) * (vel_bg - vel_obj) * (vel_bg + vel_obj)).sum((1, 2, 3)) - (
                dx_ind * ((vel_obj - vel_bg))).sum((1, 2, 3)) + sigma * lift / num_inference_steps
    kappa[i + 1] /= 2 * dsigma * guidance_scale * ((vel_obj - vel_bg) ** 2).sum((1, 2, 3))

    vf = vel_uncond + guidance_scale * ((vel_bg - vel_uncond) + kappa[i + 1][:, None, None, None] * (vel_obj - vel_bg))
    dx = 2 * dsigma * vf + noise
    latents += dx

    ll_obj[i + 1] = ll_obj[i] + (-torch.abs(dsigma) / sigma * (vel_obj) ** 2 - (dx * (vel_obj / sigma))).sum((1, 2, 3))
    ll_bg[i + 1] = ll_bg[i] + (-torch.abs(dsigma) / sigma * (vel_bg) ** 2 - (dx * (vel_bg / sigma))).sum((1, 2, 3))

get_image(latents, 1, 6)