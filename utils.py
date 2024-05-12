from PIL import Image

import torch
import torch.nn.functional as F

from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.clip.modeling_clip import _make_causal_mask, _expand_mask
from torch import autograd
import accelerate
import torch.nn as nn

from PIL import Image
import numpy as np

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def discriminator_r1_loss_accelerator(accelerator, real_pred, real_w):
    grad_real, = accelerate.gradient(
        outputs=real_pred.sum(), inputs=real_w, create_graph=True #, only_inputs=True
    )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCEWithLogitsLoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)



def discriminator_r1_loss(real_pred, real_w):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_w, create_graph=True #, only_inputs=True
    )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty

def add_noise_return_paras(
    self,
    original_samples: torch.FloatTensor,
    noise: torch.FloatTensor,
    timesteps: torch.IntTensor,
) -> torch.FloatTensor:
    # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
    alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
    timesteps = timesteps.to(original_samples.device)

    sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
    sqrt_alpha_prod = sqrt_alpha_prod.flatten()
    while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
        sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

    sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
    while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

    noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
    return noisy_samples, sqrt_alpha_prod, sqrt_one_minus_alpha_prod


def text_encoder_forward(
    text_encoder = None,
    input_ids = None,
    name_batch = None,
    attention_mask = None,
    position_ids = None,
    output_attentions = None,
    output_hidden_states = None,
    return_dict = None,
    embedding_manager = None,
    only_embedding=False,
    random_embeddings = None,
    timesteps = None,
):
    output_attentions = output_attentions if output_attentions is not None else text_encoder.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else text_encoder.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else text_encoder.config.use_return_dict

    if input_ids is None:
        raise ValueError("You have to specify either input_ids")

    input_shape = input_ids.size()
    input_ids = input_ids.view(-1, input_shape[-1])
    
    hidden_states, other_return_dict = text_encoder.text_model.embeddings(input_ids=input_ids, 
                                                                          position_ids=position_ids,
                                                                          name_batch = name_batch,
                                                                          embedding_manager=embedding_manager,
                                                                          only_embedding=only_embedding,
                                                                          random_embeddings = random_embeddings,
                                                                          timesteps = timesteps,
                                                                          )
    if only_embedding:
        return hidden_states


    causal_attention_mask = _make_causal_mask(input_shape, hidden_states.dtype, device=hidden_states.device)
    if attention_mask is not None:
        attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

    encoder_outputs = text_encoder.text_model.encoder(
        inputs_embeds=hidden_states,
        attention_mask=attention_mask,
        causal_attention_mask=causal_attention_mask,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    last_hidden_state = encoder_outputs[0]
    last_hidden_state = text_encoder.text_model.final_layer_norm(last_hidden_state)

    if text_encoder.text_model.eos_token_id == 2:
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
            input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
        ]
    else:
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
            (input_ids.to(dtype=torch.int, device=last_hidden_state.device) == text_encoder.text_model.eos_token_id)
            .int()
            .argmax(dim=-1),
        ]

    if not return_dict:
        return (last_hidden_state, pooled_output) + encoder_outputs[1:]

    return BaseModelOutputWithPooling(
        last_hidden_state=last_hidden_state,
        pooler_output=pooled_output,
        hidden_states=encoder_outputs.hidden_states,
        attentions=encoder_outputs.attentions,
    )[0], other_return_dict  


def downsampling(img: torch.tensor, w: int, h: int) -> torch.tensor:
    return F.interpolate(
        img.unsqueeze(0).unsqueeze(1),
        size=(w, h),
        mode="bilinear",
        align_corners=True,
    ).squeeze()


def image_grid(images, rows=2, cols=2):
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))

    for i, img in enumerate(images):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def latents_to_images(vae, latents, scale_factor=0.18215):
    """
    Decode latents to PIL images.
    """
    scaled_latents = 1.0 / scale_factor * latents.clone()
    images = vae.decode(scaled_latents).sample
    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.detach().cpu().permute(0, 2, 3, 1).numpy()

    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def merge_and_save_images(output_images):
    image_size = output_images[0].size

    merged_width = len(output_images) * image_size[0]
    merged_height = image_size[1]

    merged_image = Image.new('RGB', (merged_width, merged_height), (255, 255, 255))

    for i, image in enumerate(output_images):
        merged_image.paste(image, (i * image_size[0], 0))

    return merged_image

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if use_lsgan:
            self.loss = nn.MSELoss() 
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real): 
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)