import pickle
import numpy as np
from PIL import Image
import os

output_dir = 'cifar10_images'
os.makedirs(output_dir, exist_ok=True)

cifar_path = 'cifar-10-python/cifar-10-batches-py'

count = 0
for batch_num in range(1, 6):
    batch_file = f'{cifar_path}/data_batch_{batch_num}'
    print(f'Loading {batch_file}...')
    
    with open(batch_file, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    
    data = batch[b'data']
    
    for i, img_flat in enumerate(data):
        img = img_flat.reshape(3, 32, 32).transpose(1, 2, 0)
        img_pil = Image.fromarray(img.astype(np.uint8))
        img_pil.save(f'{output_dir}/img_{count:05d}.png')
        count += 1
    
    print(f'  Batch {batch_num} done. Total images: {count}')

print(f'\nDone! {count} images saved to {output_dir}/')
