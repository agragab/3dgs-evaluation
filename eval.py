import json
import arg parse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)

def load_image_as_tensor(path: str) -> torch.Tensor:
    path = Path(path)
    img = np.float32(Image.open(path).convert("RGB")) / 255.0
    img = torch.from_numpy(img)
    ## from h,w,c to c,h,w (2,0,1)
    img = img.permute(2, 0, 1)
    img = img.unsqueeze(0)
    return img

def compute_psnr(rendered: torch.Tensor, ground_truth: torch.Tensor) -> float:
    psnr = PeakSignalNoiseRatio(data_range = 1.0)
    return psnr(rendered, ground_truth)

def compute_ssim(rendered: torch.Tensor, ground_truth: torch.Tensor) -> float:
    ssim = StructuralSimilarityIndexMeasure(data_range = 1.0)
    return ssim(rendered, ground_truth)

if __name__ == "__main__":
    x = torch.ones(1, 3, 100, 100)
    y = torch.zeros(1, 3, 100, 100)

    print(f"PSNR identical: {compute_psnr(x, x):.2f} dB")
    print(f"SSIM identical: {compute_ssim(x, x):.4f}")
    print(f"PSNR different: {compute_psnr(x, y):.2f} dB")
    print(f"SSIM different: {compute_ssim(x, y):.4f}")


def evaluate_scene(
    renders_dir: str,
    ground_truth_dir: str,
) -> dict:

    renders_list = sorted(Path(renders_dir).glob("*.png"))
    ground_truth_list = sorted(Path(ground_truth_dir).glob("*.png"))

    if len(renders_list) != len(ground_truth_list):
        raise ValueError(
            f"Number of renders ({len(renders_list)}) does not match "
            f"ground truth ({len(ground_truth_list)})"
        )

    render_names = [p.name for p in renders_list]
    gt_names = [p.name for p in ground_truth_list]
    if render_names != gt_names:
        raise ValueError(f"Filenames do not match: {render_names} vs {gt_names}")
        

    running_psnr = 0
    running_ssim = 0
    psnr_list = []
    ssim_list = []
    
    for render_path, ground_truth_path in zip(renders_list, ground_truth_list):
        render = load_image_as_tensor(render_path)
        ground_truth = load_image_as_tensor(ground_truth_path)

        psnr = compute_psnr(render, ground_truth)
        ssim = compute_ssim(render, ground_truth)

        running_psnr += psnr
        running_ssim += ssim

        psnr_list.append(psnr)
        ssim_list.append(ssim)

    avg_psnr = running_psnr / len(renders_list)
    avg_ssim = running_ssim / len(renders_list)

    dictionary = {
        "psnr_mean": avg_psnr,
        "ssim_mean": avg_ssim,
        "psnr_per_view": psnr_list,
        "ssim_per_view": ssim_list,
        "num_views": len(renders_list)
    }

    return dictionary


import argparse
import json

def main():
    parser = argparse.ArgumentParser(description="Evaluate 3DGS render quality")
    parser.add_argument("--renders-dir", type=str, required=True)
    parser.add_argument("--gt-dir", type=str, required=True)
    parser.add_argument("--output-path", type=str, default="results/metrics.json")
    args = parser.parse_args()

    results = evaluate_scene(args.renders_dir, args.gt_dir)

    # save to json
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"PSNR: {results['psnr_mean']:.2f} dB")
    print(f"SSIM: {results['ssim_mean']:.4f}")
    print(f"Views evaluated: {results['num_views']}")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()