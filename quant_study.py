import csv
import torch
from metrics.psnr import PSNR
from metrics.ssim import SSIM

def quantization_study(model, hr, lr):
    psnr = PSNR()
    ssim = SSIM()

    results = []

    for q in [True, False]:
        model.quantizer.enable = q
        enc, sr = model(hr, scale=1.5)

        results.append({
            "Quantization": q,
            "PSNR": psnr(sr, hr).item(),
            "SSIM": ssim(sr, hr).item()
        })

    return results

def save_csv(results, path="quantization_study.csv"):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
