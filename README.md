## GazeFusion: Saliency-guided Image Generation

[Yunxiang Zhang](https://yunxiangzhang.github.io/), Nan Wu, [Connor Lin](https://connorzlin.com/), [Gordon Wetzstein](https://web.stanford.edu/~gordonwz/), [Qi Sun](https://qisun.me/) \
Published in ACM Transactions on Applied Perception 2024 \
Presented at ACM Symposium on Applied Perception 2024 (<span style="color: #e0144c;">Best Paper Award</span> and <span style="color: #e0144c;">Best Presentation Award</span>) \
[\[Paper\]](https://www.immersivecomputinglab.org/wp-content/uploads/2024/09/gaze_fusion_tap.pdf) [\[Project Page\]](https://www.immersivecomputinglab.org/publication/gazefusion-saliency-guided-image-generation/) [\[Video\]](https://www.youtube.com/watch?v=vFa8cyYhdD4&t=3s)

<p style="width: 90%; margin: 0 auto;">
  <img src="assets/docs/saliency-guided-image-generation.gif" width="70%" />
</p>

Diffusion models offer unprecedented image generation power given just a text prompt. While emerging approaches for controlling diffusion models have enabled users to specify the desired spatial layouts of the generated content, they cannot predict or control where viewers will pay more attention due to the complexity of human vision. Recognizing the significance of attention-controllable image generation in practical applications, we present a saliency-guided framework to incorporate the data priors of human visual attention mechanisms into the generation process. Given a user-specified viewer attention distribution, our control module conditions a diffusion model to generate images that attract viewers’ attention toward the desired regions. To assess the efficacy of our approach, we performed an eye-tracked user study and a large-scale model-based saliency analysis. The results evidence that both the cross-user eye gaze distributions and the saliency models’ predictions align with the desired attention distributions. Lastly, we outline several applications, including interactive design of saliency guidance, attention suppression in unwanted regions, and adaptive generation for varied display/viewing conditions.

## Setup
First create a dedicated conda environment:

    conda env create -f environment.yml
    conda activate gazefusion

## Data Preparation
1. Place the images that will be used for building the training dataset under the `data/raw-images/` folder;
2. Download the pre-trained [EML-Net](https://github.com/SenJia/EML-NET-Saliency) modules ([res_imagenet.pth](https://drive.google.com/open?id=1-a494canr9qWKLdm-DUDMgbGwtlAJz71), [res_places.pth](https://drive.google.com/open?id=18nRz0JSRICLqnLQtAvq01azZAsH0SEzS), and [res_decoder.pth](https://drive.google.com/open?id=1vwrkz3eX-AMtXQE08oivGMwS4lKB74sH)) for saliency map generation and place them under the `models/emlnet/` folder;
3. Build the training dataset with prompt-smap-image pairs: `python process.py`;
4. Check the command-line arguments in `process.py` for more data preparation options.

## Training
1. Download the untrained GazeFusion model from [OneDrive](https://1drv.ms/u/c/3a8968df8a027819/Ebl_aRp3SgZDl_Txil98iREBdFzwPeb7zXjcHLgUCtjW4A) and place it under the `models/` folder;
2. Train the GazeFusion model with prompt-smap-image pairs: `python train.py`;
3. Check the command-line arguments in `train.py` for more training options.

## Generation
1. Download the trained GazeFusion model from [OneDrive](https://1drv.ms/u/c/3a8968df8a027819/QRl4AorfaIkggDqHRAQAAAAAmGDlCXJjbgAYRg) (or use your own trained one) and place it under the `models/` folder;
2. Place your custom saliency map files under the `assets/smaps/` folder (or use a provided one);
3. Generate images with saliency guidance: `python generate.py --smap your_smap --prompt your_prompt`;
4. Check the command-line arguments in `generate.py` for more generation options.

## Acknowledgements
We would like to thank [Saining Xie](https://www.sainingxie.com/), [Anyi Rao](https://anyirao.com/), and [Zoya Bylinskii](http://zoyathinks.com/) for fruitful early discussion, and the authors of [Stable Diffusion](https://github.com/CompVis/stable-diffusion), [ControlNet](https://github.com/lllyasviel/ControlNet), [BLIP-2](https://arxiv.org/abs/2301.12597), [EML-Net](https://github.com/SenJia/EML-NET-Saliency), and [Text2Video-Zero](https://github.com/Picsart-AI-Research/Text2Video-Zero) for their great work, based on which GazeFusion was developed.

## Citation
If you find this work helpful to your research, please consider citing [BibTeX](assets/docs/gazefusion.bib):
```bibtex
@article{zhang2024gazefusion,
  title={GazeFusion: Saliency-guided Image Generation},
  author={Zhang, Yunxiang and Wu, Nan and Lin, Connor Z and Wetzstein, Gordon and Sun, Qi},
  journal={ACM Transactions on Applied Perception},
  year={2024},
  publisher={ACM New York, NY}
}
```