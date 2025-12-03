import torch
import os
from torchvision.utils import save_image
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

# Create folders
os.makedirs('./generated_for_fid', exist_ok=True)

# Setup model (must match your training)
model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    flash_attn = False
)

diffusion = GaussianDiffusion(
    model,
    image_size = 32,
    timesteps = 1000,
    sampling_timesteps = 250
)

trainer = Trainer(
    diffusion,
    './cifar10_images',
    train_batch_size = 64,
    train_lr = 2e-4,
    train_num_steps = 50000,
    amp = False,
    results_folder = './results_50k'
)

# Load your trained model
print("Loading model...")
trainer.load(10)  # Load model-10.pt

# Generate 10,000 images
print("Generating 10,000 images (this takes ~20-30 minutes)...")
batch_size = 64
total = 10000
count = 0

while count < total:
    with torch.no_grad():
        images = diffusion.sample(batch_size=batch_size)
    
    for img in images:
        if count < total:
            save_image(img, f'./generated_for_fid/{count:05d}.png')
            count += 1
    
    print(f"Generated {count}/{total}")

print("Done generating images!")
print("")
print("Now run this command to calculate FID:")
print("python -m pytorch_fid ./cifar10_images ./generated_for_fid")
