from saicinpainting.training.trainers import load_checkpoint
from omegaconf import OmegaConf
import yaml
import torch
import numpy as np
import cv2
from PIL import Image

import os


def load_image(fname, mode='RGB', return_orig=False):
    img = np.array(Image.open(fname).convert(mode))
    if img.ndim == 3:
        img = np.transpose(img, (2, 0, 1))
    out_img = img.astype('float32') / 255
    if return_orig:
        return out_img, img
    else:
        return out_img


def ceil_modulo(x, mod):
    if x % mod == 0:
        return x
    return (x // mod + 1) * mod


def pad_img_to_modulo(img, mod):
    channels, height, width = img.shape
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)
    return np.pad(img, ((0, 0), (0, out_height - height), (0, out_width - width)), mode='symmetric')


with open("big-lama/config.yaml", 'r') as f:
    train_config = OmegaConf.create(yaml.safe_load(f))

train_config.training_model.predict_only = True
train_config.visualizer.kind = 'noop'

model = load_checkpoint(train_config, "big-lama/models/best.ckpt", strict=False, map_location='cpu')
model.freeze()
model.to(torch.device('cpu'))

with open("configs/prediction/default.yaml", 'r') as f:
    predict_config = OmegaConf.create(yaml.safe_load(f))

datadir = "needed/"

mask_filenames = sorted(
            [os.path.join(datadir, i) for i in os.listdir(datadir) if 'mask' in i and i.endswith('.png')]
        )
img_filenames = [name.rsplit('_mask', 1)[0] + ".png" for name in mask_filenames]

with torch.no_grad():
    for i in range(len(mask_filenames)):
        image = load_image(img_filenames[i], mode='RGB')
        mask = load_image(mask_filenames[i], mode='L')
        result = dict(image=image, mask=mask[None, ...])

        result['unpad_to_size'] = result['image'].shape[1:]
        result['image'] = pad_img_to_modulo(result['image'], 8)
        result['mask'] = pad_img_to_modulo(result['mask'], 8)

        result['mask'] = torch.from_numpy(result['mask']).unsqueeze(0)
        result['image'] = torch.from_numpy(result['image']).unsqueeze(0)
        result['mask'] = (result['mask'] > 0) * 1

        result = model(result)
        cur_res = result[predict_config.out_key][0].permute(1, 2, 0).detach().cpu().numpy()
        unpad_to_size = result.get('unpad_to_size', None)

        if unpad_to_size is not None:
            orig_height, orig_width = unpad_to_size
            cur_res = cur_res[:orig_height, :orig_width]

        cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
        cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
        cv2.imwrite(
            os.path.join('out', img_filenames[i].split('/')[-1]), cur_res
        )
