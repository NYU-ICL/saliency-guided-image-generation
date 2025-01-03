import argparse
import os

import cv2
import numpy as np
import torch
from PIL import Image

from cldm.hack import disable_verbosity
from cldm.model import create_model, load_state_dict
from utils.misc_utils import set_random_seed, to_output_format

parser = argparse.ArgumentParser(description='GazeFusion: Saliency-guided Image Generation')
parser.add_argument('--log-dir', type=str, default='logs', help='path to the log folder')
parser.add_argument('--smap', type=str, default='smap1.png', help='saliency map')
parser.add_argument('--prompt', type=str, default='a sailboat on the sea', help='text prompt')
parser.add_argument('--num-samples', type=int, default=2, help='number of samples')
parser.add_argument('--batch-size', type=int, default=1, help='batch size for generation')
parser.add_argument('--seed', type=int, default=123, help='random seed')
FLAGS = parser.parse_args()


def generate(args):
    smap_path = f"assets/smaps/{args.smap}"
    sd_version = "21"
    model = create_model(f"models/cldm_v{sd_version}.yaml").cpu()
    model.load_state_dict(load_state_dict(f"models/gazefusion-sd{sd_version}.ckpt", location='cpu'))
    model.eval().cuda()
    for curr in range(0, args.num_samples, args.batch_size):
        batch_size = args.batch_size if curr+args.batch_size < args.num_samples else args.num_samples-curr
        prompt_batch = [args.prompt] * batch_size
        smap_batch = torch.from_numpy(np.tile(cv2.imread(smap_path, cv2.IMREAD_GRAYSCALE)[
            None, :, :, None], (batch_size, 1, 1, 3)).astype(np.float32) / 255.0).cuda()
        data_batch = dict(jpg=smap_batch.detach().clone(), txt=prompt_batch, hint=smap_batch)
        alpha = 0.5
        guidance_scale = 9.0
        image_batch = model.log_images(data_batch, N=batch_size, unconditional_guidance_scale=guidance_scale)
        for sample_id in range(batch_size):
            smap = to_output_format(torch.clamp(image_batch["control"][sample_id], -1., 1.))[..., 0]
            sample = to_output_format(torch.clamp(image_batch[f"samples_cfg_scale_{guidance_scale:.2f}"][sample_id], -1., 1.))
            smap_color = cv2.cvtColor(cv2.applyColorMap(smap, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
            combined = (alpha * sample.astype(np.float32) + (1 - alpha) * smap_color.astype(np.float32)).astype(np.uint8)
            Image.fromarray(sample).save(f"{args.image_dir}/sample-{curr+sample_id+1:d}.png")
            Image.fromarray(combined).save(f"{args.image_dir}/combined-{curr+sample_id+1:d}.png")


def main(args):
    disable_verbosity()
    set_random_seed(args.seed)
    args.image_dir = f"{args.log_dir}/images/generation"
    os.makedirs(args.image_dir, exist_ok=True)
    generate(args)


if __name__ == '__main__':
    main(FLAGS)
