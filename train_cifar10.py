import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

"""
DDPM Training on CIFAR-10 - FIXED VERSION
- Fixed image_size to 32 (CIFAR-10 size)
- Disabled flash_attn (kernel compatibility issue)
- Disabled amp (dtype issues)
- 50,000 training steps
"""

# Model - flash_attn MUST be False to avoid kernel error
model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    flash_attn = False       # FIXED: was True, caused kernel error
)

# Diffusion - image_size MUST be 32 for CIFAR-10
diffusion = GaussianDiffusion(
    model,
    image_size = 32,         # FIXED: was 128, CIFAR-10 is 32x32!
    timesteps = 1000,
    sampling_timesteps = 250
)

# Trainer - amp must be False
trainer = Trainer(
    diffusion,
    './cifar10_images',
    train_batch_size = 64,            # A100 can handle 64
    train_lr = 2e-4,                  # DDPM paper learning rate
    train_num_steps = 50000,          # Your target: 50k steps
    gradient_accumulate_every = 1,
    ema_decay = 0.9999,
    amp = False,                      # FIXED: was True, caused dtype error
    calculate_fid = True,
    save_and_sample_every = 5000,     # Save every 5k steps
    results_folder = './results_50k'
)

if __name__ == '__main__':
    print("=" * 60)
    print("DDPM CIFAR-10 Training - FIXED VERSION")
    print("=" * 60)
    print(f"Image size: 32x32 (CIFAR-10)")
    print(f"Training steps: 50,000")
    print(f"Batch size: 64")
    print(f"flash_attn: False")
    print(f"amp: False")
    print(f"Results: ./results_50k/")
    print("=" * 60)
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    print("Starting training...")
    print("=" * 60)
    
    trainer.train()
