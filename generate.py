import argparse
import os
import random

import cv2
import numpy as np
import torch
from PIL import Image

from cldm.model import create_model, load_state_dict
from share import *

parser = argparse.ArgumentParser(description='GazeFusion: Saliency-guided Image Generation')
parser.add_argument('--smap', type=str, default='smap1.png', help='saliency map')
parser.add_argument('--prompt', type=str, default='a sailboat on the sea', help='text prompt')
parser.add_argument('--num-samples', type=int, default=2, help='number of samples')
parser.add_argument('--batch-size', type=int, default=1, help='batch size')
parser.add_argument('--seed', type=int, default=123, help='random seed')
args = parser.parse_args()


def postprocess(image):
    image = (image + 1.0) / 2.0  # (C,H,W), -1,1 -> 0,1
    image = (image.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    assert image.shape[-1] == 3
    return image


def generate(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model = create_model(os.path.join(args.config_path)).cpu()
    model.load_state_dict(load_state_dict(args.model_path, location='cpu'))
    model.eval().cuda()
    for curr in range(0, args.num_samples, args.batch_size):
        bs = args.batch_size if curr+args.batch_size < args.num_samples else args.num_samples-curr
        prompts = [args.prompt] * bs
        smaps = torch.from_numpy(np.tile(cv2.imread(args.smap_path, cv2.IMREAD_GRAYSCALE)[
                                 None, :, :, None], (bs, 1, 1, 3)).astype(np.float32) / 255.0).cuda()
        inputs = dict(jpg=smaps.detach().clone(), txt=prompts, hint=smaps)
        alpha = 0.5
        guidance_scale = 9.0
        outputs = model.log_images(inputs, N=len(inputs['jpg']), unconditional_guidance_scale=guidance_scale)
        for id in range(bs):
            smap = postprocess(torch.clamp(outputs["control"][id].cpu(), -1., 1.))[..., 0]
            sample = postprocess(torch.clamp(outputs[f"samples_cfg_scale_{guidance_scale:.2f}"][id].cpu(), -1., 1.))
            smap_color = cv2.cvtColor(cv2.applyColorMap(smap, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
            combined = (alpha * sample.astype(np.float32) + (1 - alpha) * smap_color.astype(np.float32)).astype(np.uint8)
            Image.fromarray(sample).save(os.path.join(args.log_path, f"sample-{curr+id+1:d}.png"))
            Image.fromarray(combined).save(os.path.join(args.log_path, f"combined-{curr+id+1:d}.png"))


def main(args):
    args.smap_path = os.path.join(os.getcwd(), "smaps", args.smap)
    if not os.path.exists(args.smap_path):
        raise FileNotFoundError(f"Saliency map '{args.smap_path}' is missing")
    args.model_path = os.path.join(os.getcwd(), "models", "gazefusion-sd21.ckpt")
    args.config_path = os.path.join(os.getcwd(), "models", "cldm_v21.yaml")
    if not (os.path.exists(args.model_path) and os.path.exists(args.config_path)):
        raise FileNotFoundError(f"Model checkpoint '{args.model_path}' or config file '{args.config_path}' is missing")
    args.log_path = os.path.join(os.getcwd(), "results")
    os.makedirs(args.log_path, exist_ok=True)

    generate(args)


if __name__ == '__main__':
    main(args)
