# SpriteGen: A DDPM based Sprite generator

This repository comprises the code for a simple 16x16 sprite generator that utilizes DDPM. The backward diffusion process uses an attention-based U-Net model as the denoising model. You can find the configs for training in the config.yaml file. The model is trained for 500 epochs with the MSE loss function. The loss curve over the epochs can be found below, along with the reconstructed images at different epochs.

| Loss Variation | Generated Images at Different Epochs |
| ------------- | ------------- |
|<img src="https://github.com/mahadev1995/SpriteGen/assets/51476618/33f2d2d5-0d94-498b-83bf-9b22753aea6b.png" width="400"> | <img src="https://github.com/mahadev1995/SpriteGen/assets/51476618/747125fb-20ec-41e0-80c8-28e83dc00829.gif" width="400">  |

### Example Images
| Original Dataset | Generated Images |
| ------------- | ------------- |
|<img src="https://github.com/mahadev1995/SpriteGen/assets/51476618/326082b6-1534-4922-9f06-a95faa9394cc.png" width="512">|<img src="https://github.com/mahadev1995/SpriteGen/assets/51476618/2c1852fe-a211-4dfe-9a65-6a1131b5cec1.png" width="512">|
|<img src="https://github.com/mahadev1995/SpriteGen/assets/51476618/bec98f4c-d8e8-4354-a4c8-1091bf7d128a.png" width="512">|<img src="https://github.com/mahadev1995/SpriteGen/assets/51476618/e40d9bda-540c-483a-bbc6-7a935d987906.png" width="512">|

### Running Experiments
Before running the experiment configs can be changed in the config.yml. The experiments can be run using the following code
```Python
python train.py
```
### References
- Dataset can be found at: https://www.kaggle.com/datasets/bajajganesh/sprites-16x16-dataset
- DPPM Paper: https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf
- The code base is based on the amazing YouTube video by @outliier at https://www.youtube.com/watch?v=HoKDTa5jHvg (Check out the video explanation). The GitHub repository for the video: https://github.com/dome272/Diffusion-Models-pytorch/tree/main?tab=readme-ov-file
