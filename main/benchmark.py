#!/usr/bin/env python

import os
import cv2
import math
import torch
import numpy as np
import mmcv
from os.path import join
from tqdm import tqdm

from base.utilities import get_parser, get_logger, AverageMeter
from base.baseTrainer import load_state_dict
from models import get_model
from utils import util
from dataset.torch_bicubic import imresize

# --------------------------------------------------
# Config & device
# --------------------------------------------------
cfg = get_parser()
logger = get_logger()

USE_CUDA = torch.cuda.is_available() and len(cfg.test_gpu) > 0
device = torch.device("cuda" if USE_CUDA else "cpu")

os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in cfg.test_gpu) if USE_CUDA else ""

print("Using CUDA:", USE_CUDA)
print("CUDA device count:", torch.cuda.device_count())

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

# IMPORTANT: disable cropping for visualization
crop = False


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    logger.info(cfg)
    logger.info("=> creating model ...")

    model = get_model(cfg, logger).to(device)
    model.summary(logger, None)

    if not os.path.isfile(cfg.model_path):
        raise RuntimeError(f"No checkpoint found at {cfg.model_path}")

    checkpoint = torch.load(cfg.model_path, map_location=device)
    load_state_dict(model, checkpoint["state_dict"])
    logger.info(f"=> loaded checkpoint (epoch {checkpoint.get('epoch', 'N/A')})")

    # Only Set5 (paper demo)
    from dataset.div2k import DIV2K
    test_data = DIV2K(
        data_list=os.path.join(cfg.test_root, "list", "Set5_val.txt"),
        training=False,
        cfg=cfg
    )

    loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.workers,
        drop_last=False
    )

    scale = float(cfg.test_scale)
    logger.info(f"\n=> Dataset 'Set5' (x{scale})\n")
    test(model, loader, scale)


# --------------------------------------------------
# Test
# --------------------------------------------------
def test(model, loader, scale):
    model.eval()

    save_dir = join(cfg.save_folder, f"Set5_x{scale}")
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for i, hr in enumerate(tqdm(loader)):
            hr = hr.to(device)

            # Bicubic LR (reference)
            lr = imresize(hr, scale=1.0 / scale)

            # AIDN forward
            encoded_lr, restored_hr = model(hr, scale)

            encoded_lr = torch.clamp(encoded_lr, 0, 1)
            restored_hr = torch.clamp(restored_hr, 0, 1)

            # Convert to numpy
            hr_img = util.tensor2img(hr)
            lr_img = util.tensor2img(lr)

            # ===== PAPER-CORRECT VISUALIZATION =====
            # Encoded LR is NOT shown directly.
            # We visualize a de-quantized / clipped version.
            encoded_lr_vis = util.tensor2img(encoded_lr)
            encoded_lr_vis = np.clip(encoded_lr_vis, 0.0, 1.0)

            restored_hr_img = util.tensor2img(restored_hr)

            # Convert to uint8
            lr_uint8 = (lr_img * 255).astype(np.uint8)
            enc_lr_uint8 = (encoded_lr_vis * 255).astype(np.uint8)
            sr_uint8 = (restored_hr_img * 255).astype(np.uint8)
            hr_uint8 = (hr_img * 255).astype(np.uint8)

            name = os.path.splitext(os.path.basename(loader.dataset.imgs[i]))[0]

            # Save images (paper-style comparison)
            mmcv.imwrite(mmcv.rgb2bgr(lr_uint8), join(save_dir, f"{name}_bicubic_lr.png"))
            mmcv.imwrite(mmcv.rgb2bgr(enc_lr_uint8), join(save_dir, f"{name}_aidn_lr.png"))
            mmcv.imwrite(mmcv.rgb2bgr(sr_uint8), join(save_dir, f"{name}_aidn_sr.png"))
            mmcv.imwrite(mmcv.rgb2bgr(hr_uint8), join(save_dir, f"{name}_gt_hr.png"))


# --------------------------------------------------
if __name__ == "__main__":
    main()
