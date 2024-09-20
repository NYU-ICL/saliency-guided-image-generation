## GazeFusion: Saliency-guided Image Generation

[Yunxiang Zhang](https://yunxiangzhang.github.io/), Nan Wu, [Connor Lin](https://connorzlin.com/), [Gordon Wetzstein](https://web.stanford.edu/~gordonwz/), [Qi Sun](https://qisun.me/) \
Published in ACM Transactions on Applied Perception 2024 \
Presented at ACM Symposium on Applied Perception 2024 (<span style="color: #e0144c;">Best Paper Award</span> and <span style="color: #e0144c;">Best Presentation Award</span>) \
[\[Paper\]](https://www.immersivecomputinglab.org/wp-content/uploads/2024/09/gaze_fusion_tap.pdf) [\[Project Page\]](https://www.immersivecomputinglab.org/publication/gazefusion-saliency-guided-image-generation/) [\[Video\]](https://www.youtube.com/watch?v=vFa8cyYhdD4&t=3s)

<p style="width: 90%; margin: 0 auto;">
  <img src="docs/saliency-guided-image-generation.gif" width="70%" />
</p>

Diffusion models offer unprecedented image generation power given just a text prompt. While emerging approaches for controlling diffusion models have enabled users to specify the desired spatial layouts of the generated content, they cannot predict or control where viewers will pay more attention due to the complexity of human vision. Recognizing the significance of attention-controllable image generation in practical applications, we present a saliency-guided framework to incorporate the data priors of human visual attention mechanisms into the generation process. Given a user-specified viewer attention distribution, our control module conditions a diffusion model to generate images that attract viewers’ attention toward the desired regions. To assess the efficacy of our approach, we performed an eye-tracked user study and a large-scale model-based saliency analysis. The results evidence that both the cross-user eye gaze distributions and the saliency models’ predictions align with the desired attention distributions. Lastly, we outline several applications, including interactive design of saliency guidance, attention suppression in unwanted regions, and adaptive generation for varied display/viewing conditions.

## Inference
1. Create a dedicated Conda environment: `conda env create -f environment.yaml; conda activate gazefusion`;
2. Download the trained GazeFusion model from [OneDrive](https://1drv.ms/u/s!Ahl4AorfaIk6kYkHmGDlCXJjbgAYRg) and place it under the `models/` folder;
3. Place your custom saliency map files under the `smaps/` folder (or use a provided one);
4. Generate a few image samples with saliency guidance: `python generate.py --smap your_smap --prompt your_prompt`.

## Training
The code and data for training GazeFusion will be released soon, please stay tuned!

## Acknowledgements
We would like to thank [Saining Xie](https://www.sainingxie.com/), [Anyi Rao](https://anyirao.com/), and [Zoya Bylinskii](http://zoyathinks.com/) for fruitful early discussion, and the authors of [Stable Diffusion](https://github.com/CompVis/stable-diffusion), [ControlNet](https://github.com/lllyasviel/ControlNet), [BLIP-2](https://arxiv.org/abs/2301.12597), [EML-Net](https://github.com/SenJia/EML-NET-Saliency), and [Text2Video-Zero](https://github.com/Picsart-AI-Research/Text2Video-Zero) for their great work, based on which GazeFusion was developed.

## Citation
If you find this work useful to your research, please consider citing [BibTeX](docs/gazefusion.bib):
```bibtex
@article{zhang2024gazefusion,
  title={GazeFusion: Saliency-guided Image Generation},
  author={Zhang, Yunxiang and Wu, Nan and Lin, Connor Z and Wetzstein, Gordon and Sun, Qi},
  journal={ACM Transactions on Applied Perception},
  year={2024},
  publisher={ACM New York, NY}
}
```