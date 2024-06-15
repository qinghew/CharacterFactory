<div align="center">


## CharacterFactory: Sampling Consistent Characters with GANs for Diffusion Models

### Installation

- Requirements (Only need **2.5GB** VRAM for training): 

  ```bash
  conda create -n IDEGAN python=3.8.5
  conda activate IDEGAN
  pip install -r requirements.txt
  ```

- Download pretrained models: [Stable Diffusion v2-1_512](https://huggingface.co/stabilityai/stable-diffusion-2-1-base/).

- Set the paths of pretrained SD-2.1 models as default in the Line106 of train.py or command with 

  ```bash
  --pretrained_model_name_or_path **your SD-2.1 path**
  ```


### Train

**We have already provided the pretrained weights in `training_weight`.**

  ```bash
python train.py --experiment_name="normal_GAN"
  ```

The trained IDE-GAN model will be saved in `training_weight`.


### Test

First set **your SD-2.1 path** in the test files.

- [test.ipynb](https://github.com/qinghew/CharacterFactory/blob/main/test.ipynb): Generate a random character with text prompts. 

- [test_create_many_characters.ipynb](https://github.com/qinghew/CharacterFactory/blob/main/test_create_many_characters.ipynb): Generate many characters with text prompts.

The results will be generated in `test_results/{index}/`.
