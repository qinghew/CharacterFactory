import os
import io
import IPython.display
from PIL import Image
import base64
import io
from PIL import Image
import gradio as gr
import requests
import time
import random
import numpy as np
import torch
import os
from transformers import ViTModel, ViTImageProcessor
from utils import text_encoder_forward
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from utils import latents_to_images, downsampling, merge_and_save_images
from omegaconf import OmegaConf
from accelerate.utils import set_seed
from tqdm import tqdm
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipelineOutput
from PIL import Image
from models.celeb_embeddings import embedding_forward
import models.embedding_manager
import importlib
import time
from PIL import Image

import os
os.environ['GRADIO_TEMP_DIR'] = 'qinghewang/tmp'

title = r"""
<h1 align="center">CharacterFactory: Sampling Consistent Characters with GANs for Diffusion Models</h1>
"""

description = r"""
<b>Official Gradio demo</b> for <a href='https://qinghew.github.io/CharacterFactory/' target='_blank'><b>CharacterFactory: Sampling Consistent Characters with GANs for Diffusion Models</b></a>.<br>

How to use:<br>
1. Enter prompts (the character placeholder is "a person"), where each line will generate an image.
2. You can choose to create a new character or continue to use the current one. We have provided some examples, click on the examples below to use.
3. You can choose to use the Normal version (the gender is random), the Man version, and the Woman version.
4. Click the <b>Generate</b> button to begin (Images are generated one by one).
5. Our method can be applied to illustrating books and stories, creating brand ambassadors, developing presentations, art design, identity-consistent data construction and more. Looking forward to your explorations!😊
6. If CharacterFactory is helpful, please help to ⭐ the <a href='https://github.com/qinghew/CharacterFactory' target='_blank'>Github Repo</a>. Thanks! 
"""

article = r"""
---
📝 **Citation**
<br>
If our work is helpful for your research or applications, please cite us via:
```bibtex
@article{wang2024characterfactory,
title={CharacterFactory: Sampling Consistent Characters with GANs for Diffusion Models},
author={Wang, Qinghe and Li, Baolu and Li, Xiaomin and Cao, Bing and Ma, Liqian and Lu, Huchuan and Jia, Xu},
journal={arXiv preprint arXiv:2404.15677},
year={2024}
}
```
📧 **Contact**
<br>
If you have any questions, please feel free to open an issue or directly reach us out at <b>qinghewang@mail.dlut.edu.cn</b>.
"""

css = '''
#color-bg{display:flex;justify-content: center;align-items: center;}
.color-bg-item{width: 100%; height: 32px}
#main_button{width:100%}
<style>
'''
model_id = "stabilityai/stable-diffusion-2-1-base"
pipe = StableDiffusionPipeline.from_pretrained(model_id)   # , torch_dtype=torch.float16

# model_path = "/home/use/.cache/huggingface/hub/models--stabilityai--stable-diffusion-2-1/snapshots/5cae40e6a2745ae2b01ad92ae5043f95f23644d6"
# pipe = StableDiffusionPipeline.from_pretrained(model_path)   
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

vae = pipe.vae
unet = pipe.unet
text_encoder = pipe.text_encoder
tokenizer = pipe.tokenizer
scheduler = pipe.scheduler

input_dim = 64


original_forward = text_encoder.text_model.embeddings.forward
text_encoder.text_model.embeddings.forward = embedding_forward.__get__(text_encoder.text_model.embeddings)
embedding_manager_config = OmegaConf.load("datasets_face/identity_space.yaml")

normal_Embedding_Manager = models.embedding_manager.EmbeddingManagerId_adain(  
        tokenizer,
        text_encoder,
        device = device,
        training = True,
        experiment_name = "normal_GAN", 
        num_embeds_per_token = embedding_manager_config.model.personalization_config.params.num_embeds_per_token,            
        token_dim = embedding_manager_config.model.personalization_config.params.token_dim,
        mlp_depth = embedding_manager_config.model.personalization_config.params.mlp_depth,
        loss_type = embedding_manager_config.model.personalization_config.params.loss_type,
        vit_out_dim = input_dim,
)

man_Embedding_Manager = models.embedding_manager.EmbeddingManagerId_adain(  
        tokenizer,
        text_encoder,
        device = device,
        training = True,
        experiment_name = "man_GAN", 
        num_embeds_per_token = embedding_manager_config.model.personalization_config.params.num_embeds_per_token,            
        token_dim = embedding_manager_config.model.personalization_config.params.token_dim,
        mlp_depth = embedding_manager_config.model.personalization_config.params.mlp_depth,
        loss_type = embedding_manager_config.model.personalization_config.params.loss_type,
        vit_out_dim = input_dim,
)

woman_Embedding_Manager = models.embedding_manager.EmbeddingManagerId_adain(  
        tokenizer,
        text_encoder,
        device = device,
        training = True,
        experiment_name = "woman_GAN", 
        num_embeds_per_token = embedding_manager_config.model.personalization_config.params.num_embeds_per_token,            
        token_dim = embedding_manager_config.model.personalization_config.params.token_dim,
        mlp_depth = embedding_manager_config.model.personalization_config.params.mlp_depth,
        loss_type = embedding_manager_config.model.personalization_config.params.loss_type,
        vit_out_dim = input_dim,
)

text_encoder.text_model.embeddings.forward = original_forward

DEFAULT_STYLE_NAME = "Watercolor"
MAX_SEED = np.iinfo(np.int32).max


def replace_phrases(prompt):
    replacements = {
        "a person": "v1* v2*",
        "a man": "v1* v2*",
        "a woman": "v1* v2*",
        "a boy": "v1* v2*",
        "a girl": "v1* v2*"
    }
    for phrase, replacement in replacements.items():
        prompt = prompt.replace(phrase, replacement)
    return prompt


def handle_prompts(prompts_array):
    prompts = prompts_array.splitlines()
    prompts = [prompt + ', facing to camera, best quality, ultra high res'  for prompt in prompts]
    prompts = [replace_phrases(prompt) for prompt in prompts]
    return prompts


def generate_image(chose_emb, choice, gender_GAN, prompts_array):
    prompts = handle_prompts(prompts_array)
    
    print("gender:",gender_GAN)
    
    # if choice == "Create a new character":
    #     c = "create"
    # elif choice == "Still use this character":
    #     c = "continue"
    
    # if gender_GAN == "Normal":
    #     e = "normal_GAN"
    # elif gender_GAN == "Man":
    #     e = "man_GAN"
    # elif gender_GAN == "Woman":
    #     e = "woman_GAN"

    if gender_GAN == "Normal":
        experiment_name = "normal_GAN"
        steps = 10000
        Embedding_Manager = normal_Embedding_Manager
    elif gender_GAN == "Man":
        experiment_name = "man_GAN"
        steps = 7000
        Embedding_Manager = man_Embedding_Manager
    elif gender_GAN == "Woman":
        experiment_name = "woman_GAN"
        steps = 6000
        Embedding_Manager = woman_Embedding_Manager
    else:
        print("Hello, please notice this ^_^")
        assert 0
    
    embedding_path = os.path.join("training_weight", experiment_name, "embeddings_manager-{}.pt".format(str(steps)))
    Embedding_Manager.load(embedding_path)
    print("embedding_path:",embedding_path)
    print("choice:",choice)
    
    # index = "0"
    save_dir = os.path.join("test_results/" + gender_GAN)   # , index
    os.makedirs(save_dir, exist_ok=True)
    
    
    random_embedding = torch.randn(1, 1, input_dim).to(device)
    if choice == "Create a new character":
        _, emb_dict = Embedding_Manager(tokenized_text=None, embedded_text=None, name_batch=None, random_embeddings = random_embedding, timesteps = None,)
        test_emb = emb_dict["adained_total_embedding"].to(device)
    elif choice == "Still use this character":
        test_emb = torch.load(chose_emb).cuda()
    
    test_emb_path = os.path.join(save_dir, "id_embeddings.pt")
    torch.save(test_emb, test_emb_path)
    
    v1_emb = test_emb[:, 0]
    v2_emb = test_emb[:, 1]
    embeddings = [v1_emb, v2_emb]
    
    tokens = ["v1*", "v2*"]
    tokenizer.add_tokens(tokens)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    text_encoder.resize_token_embeddings(len(tokenizer), pad_to_multiple_of = 8)
    for token_id, embedding in zip(token_ids, embeddings):
        text_encoder.get_input_embeddings().weight.data[token_id] = embedding
    
    total_results = []
    
    i = 0
    for prompt in prompts:
        image = pipe(prompt, guidance_scale = 8.5).images
        total_results = image + total_results
        i+=1
        if i < len(prompts):
            yield total_results, gr.update(visible=True, value="<h3>(Not Finished) Generating ···</h3>"), test_emb_path
        else:
            yield total_results, gr.update(visible=True, value="<h3>Generation Finished</h3>"), test_emb_path
    
    total_results = []

def get_example():
    case = [
        [
            'demo_embeddings/example_1.pt',
            'Still use this character',
            "Normal",
            "a photo of a person\na person as a small child\na person as a 20 years old person\na person as a 80 years old person\na person reading a book\na person in the sunset\n",
        ],
        [
            'demo_embeddings/example_2.pt',
            'Still use this character',
            "Man",
            "a photo of a person\na person with a mustache and a hat\na person wearing headphoneswith red hair\na person with his dog\n",
        ],
        [
            'demo_embeddings/example_3.pt',
            'Still use this character',
            "Woman",
            "a photo of a person\na person at a beach\na person as a police officer\na person wearing a birthday hat\n",
        ],
        [
            'demo_embeddings/example_4.pt',
            'Still use this character',
            "Man",
            "a photo of a person\na person holding a bunch of flowers\na person in a lab coat\na person speaking at a podium\n",
        ],
        [
            'demo_embeddings/example_5.pt',
            'Still use this character',
            "Woman",
            "a photo of a person\na person wearing a kimono\na person in Van Gogh style\nEthereal fantasy concept art of a person\n",
        ],
        [
            'demo_embeddings/example_6.pt',
            'Still use this character',
            "Man",
            "a photo of a person\na person in the rain\na person meditating\na pencil sketch of a person\n",
        ],
    ]
    return case


with gr.Blocks(css=css) as demo:    # css=css
    # binary_matrixes = gr.State([])
    # color_layout = gr.State([])
    
    # gr.Markdown(logo)
    gr.Markdown(title)
    gr.Markdown(description)
    
    with gr.Row():
        with gr.Column():         
            prompts_array = gr.Textbox(lines = 3, 
                    label="Prompts (each line corresponds to a frame).", 
                    info="Give simple prompt is enough to achieve good face fidelity",
                    value="a photo of a person\na person reading a book\na person wearing a Christmas hat\na Fauvism painting of a person\n",
                    interactive=True)
            choice = gr.Radio(choices=["Create a new character", "Still use this character"], label="Choose your action")

            gender_GAN = gr.Radio(choices=["Normal", "Man", "Woman"], label="Choose your model version (Only work for 'Create a new character')")   # , disabled=False
            chose_emb = gr.File(label="Uploaded files", type="filepath", visible=False)
            
            generate = gr.Button("Generate!😊", variant="primary")

        with gr.Column():
            gallery = gr.Gallery(label="Generated Images", columns=2, height='auto')
            generated_information = gr.Markdown(label="Generation Details", value="",visible=False)
            
        generate.click(
            fn=generate_image,
            inputs=[chose_emb, choice, gender_GAN, prompts_array],
            outputs=[gallery, generated_information, chose_emb] 
        )
        
        

    gr.Examples(
        examples=get_example(),
        inputs=[chose_emb, choice, gender_GAN, prompts_array],
        run_on_click=False,
        fn=generate_image,
        outputs=[gallery, generated_information, chose_emb],
    )
    
    gr.Markdown(article)

demo.launch()   # share=True