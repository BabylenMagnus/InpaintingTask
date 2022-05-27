from model import BaseModel
from omegaconf import OmegaConf
import yaml
import torch
import os
from utils import load_image, pad_img_to_modulo
import numpy as np
import cv2


with open("config.yaml", 'r') as f:
    train_config = OmegaConf.create(yaml.safe_load(f))

train_config.visualizer.kind = 'noop'

kwargs = dict(train_config.training_model)
kwargs.pop('kind')
kwargs.pop('store_discr_outputs_for_vis')
kwargs['use_ddp'] = train_config.trainer.kwargs.get('accelerator', None) == 'ddp'

alg = BaseModel(train_config, **kwargs)
state = torch.load("best.ckpt", map_location="cpu")
alg.load_state_dict(state['state_dict'], strict=False)
alg.on_load_checkpoint(state)

alg.freeze()
alg.to(torch.device('cpu'))

datadir = "images"

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

        result = alg(result)
        cur_res = result['inpainted'][0].permute(1, 2, 0).detach().cpu().numpy()
        unpad_to_size = result.get('unpad_to_size', None)

        if unpad_to_size is not None:
            orig_height, orig_width = unpad_to_size
            cur_res = cur_res[:orig_height, :orig_width]

        cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
        cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
        cv2.imwrite(
            os.path.join('out', img_filenames[i].split('/')[-1]), cur_res
        )
