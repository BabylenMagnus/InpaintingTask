#!/usr/bin/env python3

# Example command:
# ./bin/predict.py \
#       model.path=<path to checkpoint, prepared by make_checkpoint.py> \
#       indir=<path to input data> \
#       outdir=<where to store predicts>

import logging
import os
import sys
import traceback

from saicinpainting.evaluation.utils import move_to_device

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import cv2
import numpy as np
import torch
import tqdm
import yaml
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate

from saicinpainting.training.data.datasets import make_default_val_dataset
from saicinpainting.utils import register_debug_signal_handlers
from saicinpainting.training.trainers import load_checkpoint

LOGGER = logging.getLogger(__name__)


def main():
    try:
        with open("configs/prediction/default.yaml", 'r') as f:
            predict_config = OmegaConf.create(yaml.safe_load(f))

        register_debug_signal_handlers()  # kill -10 <pid> will result in traceback dumped into log

        device = torch.device('cpu')

        with open("big-lama/config.yaml", 'r') as f:
            train_config = OmegaConf.create(yaml.safe_load(f))
        
        train_config.training_model.predict_only = True
        train_config.visualizer.kind = 'noop'

        out_ext = predict_config.get('out_ext', '.png')

        model = load_checkpoint(train_config, "big-lama/models/best.ckpt", strict=False, map_location='cpu')

        model.freeze()
        model.to(device)

        dataset = make_default_val_dataset("needed/", **predict_config.dataset)
        with torch.no_grad():
            for img_i in tqdm.trange(len(dataset)):
                mask_fname = dataset.mask_filenames[img_i]
                cur_out_fname = os.path.join(
                    'out',
                    os.path.splitext(mask_fname[len("needed/"):])[0] + out_ext
                )
                os.makedirs(os.path.dirname(cur_out_fname), exist_ok=True)
                batch = move_to_device(default_collate([dataset[img_i]]), device)
                print(batch.keys())
                batch['mask'] = (batch['mask'] > 0) * 1
                batch = model(batch)
                cur_res = batch[predict_config.out_key][0].permute(1, 2, 0).detach().cpu().numpy()

                unpad_to_size = batch.get('unpad_to_size', None)
                if unpad_to_size is not None:
                    orig_height, orig_width = unpad_to_size
                    cur_res = cur_res[:orig_height, :orig_width]

                cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
                cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
                cv2.imwrite(cur_out_fname, cur_res)
    except KeyboardInterrupt:
        LOGGER.warning('Interrupted by user')
    except Exception as ex:
        LOGGER.critical(f'Prediction failed due to {ex}:\n{traceback.format_exc()}')
        sys.exit(1)


if __name__ == '__main__':
    main()
