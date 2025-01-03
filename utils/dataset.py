import json
import os

import cv2
import numpy as np
from torch.utils.data import Dataset


class GazeFusionDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data = []
        with open(f"{self.data_dir}/prompts/prompts.json", 'rt') as file:
            for line in file:
                info = json.loads(line)
                filename = info["filename"]
                image_path = f"{self.data_dir}/images/{filename}"
                smap_path = f"{self.data_dir}/smaps/{filename}"
                if os.path.isfile(image_path) and os.path.isfile(smap_path):
                    data_entry = {"image": image_path, "smap": smap_path, "prompt": info["prompt"]}
                    self.data.append(data_entry)
        if len(self.data) == 0:
            raise ValueError(f"No data found at {data_dir}")
        print("***************************************")
        print(f"GazeFusion dataset loaded: {len(self.data):d} prompt-smap-image pairs", flush=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_entry = self.data[idx]
        image = (cv2.cvtColor(cv2.imread(data_entry["image"], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5) - 1.0
        smap = np.tile(np.expand_dims(cv2.imread(data_entry["smap"], cv2.IMREAD_GRAYSCALE), 2), (1, 1, 3)).astype(np.float32) / 255.0
        prompt = data_entry["prompt"]
        return dict(jpg=image, txt=prompt, hint=smap)


if __name__ == '__main__':
    dataset = GazeFusionDataset("data")
