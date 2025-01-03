import argparse
import json
import os

os.environ['HF_HOME'] = "models/huggingface"
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = "1"

import numpy as np
import torch
from PIL import Image
from torchvision.io import ImageReadMode
from torchvision.io.image import read_image
from torchvision.transforms.functional import resize
from transformers import Blip2ForConditionalGeneration, Blip2Processor

from models.emlnet import decoder, resnet
from utils.misc_utils import gaussian_smooth, set_random_seed

parser = argparse.ArgumentParser(description='GazeFusion: Saliency-guided Image Generation')
parser.add_argument('--data-dir', type=str, default='data', help='path to the data folder')
parser.add_argument('--resolution', type=int, default=512, help='resolution of the (square) images prepared for training')
parser.add_argument('--smap-batch-size', type=int, default=128, help='batch size for saliency prediction')
parser.add_argument('--prompt-batch-size', type=int, default=128, help='batch size for image captioning')
parser.add_argument('--seed', type=int, default=123, help='random seed')
FLAGS = parser.parse_args()


def prepare_images(args):
    print("***************************************")
    print("Start image preparation...", flush=True)
    stride = args.resolution // 2
    for filename in sorted(os.listdir(args.raw_image_dir)):
        image = read_image(f"{args.raw_image_dir}/{filename}", mode=ImageReadMode.RGB)
        image = resize(image, args.resolution)
        height, width = image.shape[1:]
        landscape = height <= width
        for patch_id in range((max(height, width) - min(height, width)) // stride + 1):
            if landscape:
                patch = image[..., patch_id*stride:patch_id*stride+args.resolution]
            else:
                patch = image[:, patch_id*stride:patch_id*stride+args.resolution]
            patch = patch.numpy().transpose(1, 2, 0)
            Image.fromarray(patch).save(f"{args.image_dir}/{filename.split('.')[0]}_{patch_id+1:d}.jpg")
        if (max(height, width) - min(height, width)) % stride >= stride // 2:
            patch = image[..., -args.resolution:] if landscape else image[:, -args.resolution:]
            patch = patch.numpy().transpose(1, 2, 0)
            Image.fromarray(patch).save(f"{args.image_dir}/{filename.split('.')[0]}_{patch_id+2:d}.jpg")
    print("Image preparation completed", flush=True)


def prepare_smaps(args):
    print("***************************************")
    print("Start saliency map preparation...", flush=True)
    num_features = 5
    sod_resolution = (480, 640)
    imagenet_model = resnet.resnet50("models/emlnet/res_imagenet.pth").cuda().eval()
    places_model = resnet.resnet50("models/emlnet/res_places.pth").cuda().eval()
    decoder_model = decoder.build_decoder("models/emlnet/res_decoder.pth", sod_resolution, num_features, num_features).cuda().eval()

    filename_list = sorted(os.listdir(args.image_dir))
    image_batch = []
    filename_batch = []
    for file_id, filename in enumerate(filename_list, 1):
        if os.path.exists(f"{args.smap_dir}/{filename}"):
            continue
        image = read_image(f"{args.image_dir}/{filename}", mode=ImageReadMode.RGB).float()
        if image.mean().item() > 1.0:
            image_batch.append(image)
            filename_batch.append(filename)
        if len(image_batch) >= args.smap_batch_size or file_id == len(filename_list):
            image_batch = resize(torch.stack(image_batch, dim=0) / 255.0, sod_resolution).cuda()
            with torch.no_grad():
                imagenet_features = imagenet_model(image_batch, decode=True)
                places_features = places_model(image_batch, decode=True)
                smap_batch = decoder_model([imagenet_features, places_features])
            smap_batch = resize(smap_batch.squeeze(dim=1).detach().cpu(), (args.resolution, args.resolution))
            for filename, smap in zip(filename_batch, smap_batch):
                smap = (gaussian_smooth(smap.numpy()) * 255.0).astype(np.uint8)
                Image.fromarray(smap).save(f"{args.smap_dir}/{filename}")
            image_batch = []
            filename_batch = []
    print("Saliency map preparation completed", flush=True)


def prepare_prompts(args):
    print("***************************************")
    print("Start text prompt preparation...", flush=True)
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b-coco")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b-coco", torch_dtype=torch.float16).cuda()
    file = open(f"{args.prompt_dir}/prompts.json", 'w')

    filename_list = sorted(os.listdir(args.image_dir))
    image_batch = []
    filename_batch = []
    for file_id, filename in enumerate(filename_list, 1):
        image = Image.open(f"{args.image_dir}/{filename}").convert("RGB")
        image_batch.append(image)
        filename_batch.append(filename)
        if len(image_batch) >= args.prompt_batch_size or file_id == len(filename_list):
            image_batch = processor(image_batch, return_tensors="pt").to("cuda", torch.float16)
            prompt_batch = processor.batch_decode(model.generate(**image_batch, max_new_tokens=20), skip_special_tokens=True)
            for filename, prompt in zip(filename_batch, prompt_batch):
                entry = {"filename": filename, "prompt": prompt.strip()}
                json.dump(entry, file)
                file.write('\n')
            file.flush()
            image_batch = []
            filename_batch = []
    file.close()
    print("Text prompt preparation completed", flush=True)


def process(args):
    set_random_seed(args.seed)
    prepare_images(args)
    prepare_smaps(args)
    prepare_prompts(args)


def main(args):
    args.raw_image_dir = f"{args.data_dir}/raw-images"
    args.image_dir = f"{args.data_dir}/images"
    args.smap_dir = f"{args.data_dir}/smaps"
    args.prompt_dir = f"{args.data_dir}/prompts"
    if not os.path.isdir(args.raw_image_dir):
        raise FileNotFoundError(f"Raw image folder '{args.raw_image_dir}' is missing")
    os.makedirs(args.image_dir, exist_ok=True)
    os.makedirs(args.smap_dir, exist_ok=True)
    os.makedirs(args.prompt_dir, exist_ok=True)
    process(args)


if __name__ == '__main__':
    main(FLAGS)
