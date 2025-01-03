import argparse
import os

os.environ['HF_HOME'] = "models/huggingface"
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = "1"

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from cldm.hack import disable_verbosity
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from utils.dataset import GazeFusionDataset
from utils.misc_utils import set_random_seed

parser = argparse.ArgumentParser(description='GazeFusion: Saliency-guided Image Generation')
parser.add_argument('--data-dir', type=str, default='data', help='path to the data folder')
parser.add_argument('--log-dir', type=str, default='logs', help='path to the log folder')
parser.add_argument('--log-freq', type=int, default=1000, help='logging frequency')
parser.add_argument('--ckpt-freq', type=int, default=5000, help='checkpointing frequency')
parser.add_argument('--ckpt-topk', type=int, default=1, help='only keep the k latest checkpoints')
parser.add_argument('--max-steps', type=int, default=round(5e5), help='number of training steps')
parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
parser.add_argument('--batch-size', type=int, default=16, help='batch size for training')
parser.add_argument('--unlock-sd', action="store_true", default=False, help='unlock the stable diffusion model for training')
parser.add_argument('--mid-control', action="store_true", default=False, help='only feed control signals to the middle block')
parser.add_argument('--seed', type=int, default=123, help='random seed')
FLAGS = parser.parse_args()


def main(args):
    disable_verbosity()
    set_random_seed(args.seed)
    # Data and logging
    dataset = GazeFusionDataset(args.data_dir)
    dataloader = DataLoader(dataset, num_workers=8, batch_size=args.batch_size, shuffle=True)
    os.makedirs(args.log_dir, exist_ok=True)
    args.ckpt_dir = f"{args.log_dir}/checkpoints"
    logger = ImageLogger(args.log_dir, batch_frequency=args.log_freq)
    # Model
    sd_version = "21"
    model = create_model(f"models/cldm_v{sd_version}.yaml").cpu()
    model.load_state_dict(load_state_dict(f"models/gazefusion-init-sd{sd_version}.ckpt", location='cpu'))
    model.learning_rate, model.sd_locked, model.only_mid_control = args.lr, not args.unlock_sd, args.mid_control
    # Training
    checkpoint_callback = ModelCheckpoint(dirpath=args.ckpt_dir, filename=f"gazefusion-sd{sd_version}-"+"{global_step}", monitor="global_step",
                                          save_top_k=args.ckpt_topk, mode='max', every_n_train_steps=args.ckpt_freq)
    trainer = pl.Trainer(accelerator="gpu", precision=32, callbacks=[logger, checkpoint_callback], max_steps=args.max_steps)
    trainer.fit(model, dataloader)


if __name__ == '__main__':
    main(FLAGS)
