import argparse
import itertools
import logging
import math
import os
from pathlib import Path
import accelerate
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from packaging import version
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    UNet2DConditionModel,
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
import numpy as np
from omegaconf import OmegaConf
import random
from transformers import ViTModel, ViTImageProcessor
from models.celeb_embeddings import embedding_forward  
from models.embedding_manager import EmbeddingManagerId_adain, Embedding_discriminator
from datasets_face.face_id import FaceIdDataset
from utils import text_encoder_forward, set_requires_grad, add_noise_return_paras, latents_to_images, discriminator_r1_loss, discriminator_r1_loss_accelerator, downsampling, GANLoss
import types
import torch.nn as nn
from tqdm import tqdm
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipelineOutput
import importlib

logger = get_logger(__name__)

def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a script for training Cones 2.")          
    parser.add_argument(
        "--embedding_manager_config", 
        type=str,
        default="datasets_face/identity_space.yaml",
        help=('config to load the train model and dataset'),
    )
    parser.add_argument(
        "--d_reg_every",  
        type=int, 
        default=16,
        help="interval for applying r1 regularization"
    )    
    parser.add_argument(
        "--r1", 
        type=float, 
        default=1, 
        help="weight of the r1 regularization"
    )    
    parser.add_argument(
        "--l_gan_lambda",  
        type=float,
        default=1,
        help="Initial learning rate (after the potential warmup period) to use.",
    )  
    parser.add_argument(
        "--l_consis_lambda",  
        type=float,
        default=8,
        help="Initial learning rate (after the potential warmup period) to use.",
    )                             
    parser.add_argument(
        "--pretrained_model_name_or_path", 
        type=str,
        default="/home/user/.cache/huggingface/hub/models--stabilityai--stable-diffusion-2-1/snapshots/5cae40e6a2745ae2b01ad92ae5043f95f23644d6",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_embedding_manager_path",  
        type=str,
        default=None,          
        help="pretrained_embedding_manager_path",
    )      
    parser.add_argument(
        "--pretrained_embedding_manager_epoch",  
        type=str,
        default=800,
        help="pretrained_embedding_manager_epoch",
    )                
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be"
            " float32 precision."
        ),
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )      
    parser.add_argument(
        "--output_dir",
        type=str,
        default="training_weight/normal_GAN",   # training_weight/woman_GAN  training_weight/man_GAN
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default= None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--train_batch_size", 
        type=int, default=8, 
        help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--num_train_epochs", 
        type=int, 
        default=None
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        # default=None,
        default=10001,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=1000,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via"
            " `--resume_from_checkpoint`. In the case that the checkpoint is better than the final trained model, the"
            " checkpoint can also be used for inference. Using a checkpoint for inference requires separate loading of"
            " the original pipeline and the individual checkpointed model components."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,          
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=2,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--input_dim",   
        type=int, 
        default=64,
        help="randomly sampled vectors and dimensions of MLP input"
    )
    parser.add_argument(
        "--experiment_name",   
        type=str, 
        default="normal_GAN",  # "man_GAN"  "woman_GAN"
        help="randomly sampled vectors and dimensions of MLP input"
    )
    
    
    if input_args is not None:  
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

def encode_prompt(prompt_batch, name_batch, text_encoder, tokenizer, embedding_manager, is_train=True,
                  random_embeddings = None, timesteps = None):
    captions = []
    proportion_empty_prompts = 0     
    
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")  
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            captions.append(random.choice(caption) if is_train else caption[0])

    text_inputs = tokenizer(
        captions,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids.to(text_encoder.device) 
    
    positions_list = []
    for prompt_ids in text_input_ids:
        position = int(torch.where(prompt_ids == 265)[0][0])
        positions_list.append(position)

    prompt_embeds, other_return_dict = text_encoder_forward(
                                            text_encoder = text_encoder, 
                                            input_ids = text_input_ids,
                                            name_batch = name_batch,
                                            output_hidden_states=True,
                                            embedding_manager = embedding_manager,
                                            random_embeddings = random_embeddings,
                                            timesteps = timesteps)

    return prompt_embeds, other_return_dict, positions_list


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:  
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)  
        torch.nn.init.constant_(m.bias.data, 0.0)   


def main(args):
    args.output_dir = os.path.join(args.output_dir, args.experiment_name)
    print("output_dir", args.output_dir)    
    logging_dir = Path(args.output_dir, args.logging_dir) 
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    if args.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizer  
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )
    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

    # Load scheduler and models  
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    noise_scheduler.add_noise = types.MethodType(add_noise_return_paras, noise_scheduler)
    
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )

    text_encoder.text_model.embeddings.forward = embedding_forward.__get__(text_encoder.text_model.embeddings)

    
    embedding_manager_config = OmegaConf.load(args.embedding_manager_config)
    experiment_name = args.experiment_name

    Embedding_Manager = EmbeddingManagerId_adain(   
            tokenizer,
            text_encoder,
            device = accelerator.device,
            training = True,
            num_embeds_per_token = embedding_manager_config.model.personalization_config.params.num_embeds_per_token,            
            token_dim = embedding_manager_config.model.personalization_config.params.token_dim,
            mlp_depth = embedding_manager_config.model.personalization_config.params.mlp_depth,
            loss_type = embedding_manager_config.model.personalization_config.params.loss_type,
            input_dim = embedding_manager_config.model.personalization_config.params.input_dim,
            experiment_name = experiment_name,
    )
    
    Embedding_Manager.name_projection_layer.apply(weights_init_normal)
    
    Embedding_D = Embedding_discriminator(embedding_manager_config.model.personalization_config.params.token_dim * 2, dropout_rate = 0.2)
    Embedding_D.apply(weights_init_normal)
    
    if args.pretrained_embedding_manager_path is not None:
        epoch = args.pretrained_embedding_manager_epoch
        embedding_manager_path = os.path.join(args.pretrained_embedding_manager_path, "embeddings_manager-{}.pt".format(epoch))
        Embedding_Manager.load(embedding_manager_path)
        embedding_D_path = os.path.join(args.pretrained_embedding_manager_path, "embedding_D-{}.pt".format(epoch))        
        Embedding_D = torch.load(embedding_D_path)
    
    for param in Embedding_Manager.trainable_projection_parameters():
        param.requires_grad = True
    Embedding_D.requires_grad = True
        
    text_encoder.requires_grad_(False)


    # Check that all trainable models are in full precision
    low_precision_error_string = (
        "Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training. copy of the weights should still be float32."
    )

    if accelerator.unwrap_model(text_encoder).dtype != torch.float32:
        raise ValueError(
            f"Text encoder loaded as datatype {accelerator.unwrap_model(text_encoder).dtype}."
            f" {low_precision_error_string}"
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW


    projection_params_to_optimize = Embedding_Manager.trainable_projection_parameters()
    optimizer_projection = optimizer_class(
        projection_params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    discriminator_params_to_optimize = list(Embedding_D.parameters())
    optimizer_discriminator = optimizer_class(
        discriminator_params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )


    train_dataset = FaceIdDataset(
        experiment_name = experiment_name
    )
    
    print("dataset_length", train_dataset._length)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=accelerator.num_processes,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    
    lr_scheduler_proj = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer_projection,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    lr_scheduler_disc = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer_discriminator,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )    
    
    Embedding_Manager, optimizer_projection, optimizer_discriminator, train_dataloader, lr_scheduler_proj, lr_scheduler_disc = accelerator.prepare(
        Embedding_Manager, optimizer_projection, optimizer_discriminator, train_dataloader, lr_scheduler_proj, lr_scheduler_disc
    )


    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae and unet to device and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    Embedding_Manager.to(accelerator.device, dtype=weight_dtype)
    Embedding_D.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("identity_space", config=vars(args))
    
    
    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    num_iter = 0 
    # trained_images_num = 0
    for epoch in range(first_epoch, args.num_train_epochs): 
        print("=====================================")
        print("epoch:", epoch)
        print("=====================================")
        Embedding_Manager.train()  
        for step, batch in enumerate(train_dataloader):  
                 
            # Skip steps until we reach the resumed step 
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            random_embeddings = torch.randn(1, 1, args.input_dim).to(accelerator.device)
            random_embeddings = random_embeddings.repeat(args.train_batch_size, 1, 1)
            
            encoder_hidden_states, other_return_dict, positions_list = encode_prompt(batch["caption"], 
                                                                                    batch["name"], 
                                                                                    text_encoder, tokenizer, 
                                                                                    Embedding_Manager, 
                                                                                    is_train=True, 
                                                                                    random_embeddings = random_embeddings, 
                                                                                    timesteps = 0)

            name_embeddings = other_return_dict["name_embeddings"]    
            adained_total_embedding = other_return_dict["adained_total_embedding"]
            fake_emb = adained_total_embedding
            
            criterionGAN = GANLoss().to(accelerator.device)
            
            set_requires_grad(Embedding_D, True)
            optimizer_discriminator.zero_grad(set_to_none=args.set_grads_to_none)
            # fake            
            pred_fake = Embedding_D(fake_emb.detach()) 
            loss_D_fake = criterionGAN(pred_fake[0], False)                   
            
            # Real
            random_noise = torch.rand_like(name_embeddings) * 0.005   
            real_name_embeddings = random_noise + name_embeddings
            pred_real = Embedding_D(real_name_embeddings)  
            loss_D_real = criterionGAN(pred_real[0], True)             
            
            loss_D = (loss_D_fake + loss_D_real) * 0.5
            accelerator.backward(loss_D)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(discriminator_params_to_optimize, args.max_grad_norm)   
            optimizer_discriminator.step()

            set_requires_grad(Embedding_D, False)  
            optimizer_projection.zero_grad(set_to_none=args.set_grads_to_none)
            pred_fake = Embedding_D(fake_emb)
             
            loss_G_GAN = criterionGAN(pred_fake[0], True) 
            
            num_embeddings = encoder_hidden_states.size(0)   
            loss_consistency = 0.0
            for i in range(num_embeddings):
                position1 = positions_list[i]
                name_embedding1 = torch.cat([encoder_hidden_states[i][position1], encoder_hidden_states[i][position1 + 1]], dim=0)
                for j in range(i + 1, num_embeddings):
                    position2 = positions_list[j]
                    name_embedding2 = torch.cat([encoder_hidden_states[j][position2], encoder_hidden_states[j][position2 + 1]], dim=0)
                    loss_consistency += F.mse_loss(name_embedding1, name_embedding2)
            
            loss_consistency /= (num_embeddings * (num_embeddings - 1)) / 2
            
            loss = loss_G_GAN * args.l_gan_lambda + loss_consistency * args.l_consis_lambda
            
            accelerator.backward(loss)  
            
            if accelerator.sync_gradients: 
                accelerator.clip_grad_norm_(projection_params_to_optimize, args.max_grad_norm)
            optimizer_projection.step()
            lr_scheduler_proj.step()
            lr_scheduler_disc.step()

            num_iter += 1

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"embeddings_manager-{global_step}.pt")
                        # accelerator.save_state(save_path)
                        try:
                            Embedding_Manager.save(save_path)
                        except:
                            Embedding_Manager.module.save(save_path)
                        
                        save_path_d = os.path.join(args.output_dir, f"embedding_D-{global_step}.pt")
                        Embedding_D.save(save_path_d)

                        logger.info(f"Saved state to {save_path}")  
                    
                global_step += 1

            adained_total_embeddings_max_min = (round(adained_total_embedding.max().detach().item(), 4),
                                               round(adained_total_embedding.min().detach().item(), 4))  
            
            logs = {"m1": adained_total_embeddings_max_min,
                    "l_G_GAN": loss_G_GAN.detach().item(),
                    "l_consistency": loss_consistency.detach().item(),
                    "l_D_real": loss_D_real.detach().item(),
                    "l_D_fake": loss_D_fake.detach().item(),                     
                    "loss": loss.detach().item(), 
                    }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            
            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
