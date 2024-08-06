import os
import math
import wandb
import random
import logging
import inspect
import argparse
import datetime
import subprocess

from pathlib import Path
from tqdm.auto import tqdm
from einops import rearrange
from omegaconf import OmegaConf
from safetensors import safe_open
from typing import Dict, Optional, Tuple

import torch
import torchvision
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim.swa_utils import AveragedModel
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import argparse, os, sys, glob, yaml, math, random
import datetime, time
import numpy as np
from omegaconf import OmegaConf
from collections import OrderedDict
from tqdm import trange, tqdm
from einops import repeat
from einops import rearrange, repeat
from functools import partial
import torch
from pytorch_lightning import seed_everything

from funcs import load_model_checkpoint, load_prompts, load_image_batch, get_filelist, save_videos
from funcs import batch_ddim_sampling
from utils.utils import instantiate_from_config

from scripts.evaluation.data.dataset import WebVid10M
import diffusers
from diffusers.optimization import get_scheduler

from utils.utils import save_videos_grid



def get_parser():
    parser = argparse.ArgumentParser()
    
    ##########################################################################################################################################################
    
    parser.add_argument("--seed", type=int, default=20230211, help="seed for seed_everything")
    parser.add_argument("--mode", default="base", type=str, help="which kind of inference mode: {'base', 'i2v'}")
    parser.add_argument("--ckpt_path", type=str, default=None, help="checkpoint path")
    parser.add_argument("--config", type=str, help="config (yaml) path for VideoCrafter2 Model")
    parser.add_argument("--config2", type=str, help="config (yaml) path for training parameters in Animatediff model")
    parser.add_argument("--prompt_file", type=str, default=None, help="a text file containing many prompts")
    parser.add_argument("--savedir", type=str, default=None, help="results saving path")
    parser.add_argument("--savefps", type=str, default=10, help="video fps to generate")
    parser.add_argument("--n_samples", type=int, default=1, help="num of samples per prompt",)
    parser.add_argument("--ddim_steps", type=int, default=50, help="steps of ddim if positive, otherwise use DDPM",)
    parser.add_argument("--ddim_eta", type=float, default=1.0, help="eta for ddim sampling (0.0 yields deterministic sampling)",)
    parser.add_argument("--bs", type=int, default=1, help="batch size for inference")
    parser.add_argument("--height", type=int, default=512, help="image height, in pixel space")
    parser.add_argument("--width", type=int, default=512, help="image width, in pixel space")
    parser.add_argument("--frames", type=int, default=-1, help="frames num to inference")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--unconditional_guidance_scale", type=float, default=1.0, help="prompt classifier-free guidance")
    parser.add_argument("--unconditional_guidance_scale_temporal", type=float, default=None, help="temporal consistency guidance")
    ## for conditional i2v only
    parser.add_argument("--cond_input", type=str, default=None, help="data dir of conditional input")
    
    ############################################################################################################################################################
    
    ############################################################################################################################################################

    parser.add_argument("--image_finetune", type=bool, default=False, help="Flag for image fine-tuning")
    
    parser.add_argument("--name", type=str, help="Name of the run")
    parser.add_argument("--wandb",    action="store_true")
    parser.add_argument("--launcher", type=str, choices=["pytorch", "slurm"], default="pytorch")
    
    parser.add_argument("--output_dir", type=str, help="Directory for output files")
    parser.add_argument("--pretrained_model_path", type=str, help="Path to the pretrained model")
    
    parser.add_argument("--train_data", type=dict, help="Training data configuration")
    parser.add_argument("--validation_data", type=dict, help="Validation data configuration")
    parser.add_argument("--cfg_random_null_text", type=bool, default=True, help="Flag to enable random null text for CFG")
    parser.add_argument("--cfg_random_null_text_ratio", type=float, default=0.1, help="Ratio for random null text in CFG")
    
    parser.add_argument("--unet_checkpoint_path", type=str, default="", help="Path to the UNet checkpoint")
    parser.add_argument("--unet_additional_kwargs", type=dict, default={}, help="Additional keyword arguments for UNet")
    parser.add_argument("--ema_decay", type=float, default=0.9999, help="Decay rate for EMA")
    parser.add_argument("--noise_scheduler_kwargs", type=dict, default=None, help="Keyword arguments for noise scheduler")
    
    parser.add_argument("--max_train_epoch", type=int, default=-1, help="Maximum number of training epochs")
    parser.add_argument("--max_train_steps", type=int, default=100, help="Maximum number of training steps")
    parser.add_argument("--validation_steps", type=int, default=100, help="Number of validation steps")
    parser.add_argument("--validation_steps_tuple", type=tuple, default=(-1,), help="Tuple for validation steps")
    
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate for training")
    parser.add_argument("--scale_lr", type=bool, default=False, help="Flag to scale learning rate")
    parser.add_argument("--lr_warmup_steps", type=int, default=0, help="Number of learning rate warmup steps")
    parser.add_argument("--lr_scheduler", type=str, default="constant", help="Learning rate scheduler type")
    
    parser.add_argument("--trainable_modules", type=tuple, default=(None,), help="Tuple of trainable modules")
    parser.add_argument("--num_workers", type=int, default=32, help="Number of workers for data loading")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Beta1 parameter for Adam optimizer")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="Beta2 parameter for Adam optimizer")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay for Adam optimizer")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon parameter for Adam optimizer")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm for clipping")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps")
    parser.add_argument("--gradient_checkpointing", type=bool, default=False, help="Flag for gradient checkpointing")
    parser.add_argument("--checkpointing_epochs", type=int, default=5, help="Number of epochs between checkpoints")
    parser.add_argument("--checkpointing_steps", type=int, default=-1, help="Number of steps between checkpoints")
    
    parser.add_argument("--mixed_precision_training", type=bool, default=False, help="Flag for mixed precision training")
    parser.add_argument("--enable_xformers_memory_efficient_attention", type=bool, default=True, help="Flag to enable xformers memory efficient attention")
    
    parser.add_argument("--global_seed", type=int, default=42, help="Global random seed")
    parser.add_argument("--is_debug", type=bool, default=False, help="Flag for debug mode")
    
    ############################################################################################################################################################
    return parser


def train_mmg(args, gpu_num, gpu_no, **kwargs):
    
    
    
    ## step 1: model config
    ## -----------------------------------------------------------------
    config = OmegaConf.load(args.config)
    #data_config = config.pop("data", OmegaConf.create())
    model_config = config.pop("model", OmegaConf.create())

    mmg_model = instantiate_from_config(model_config)
    teacher_model = instantiate_from_config(model_config)

    mmg_model = mmg_model.cuda(gpu_no)
    teacher_model = teacher_model.cuda(gpu_no)
    assert os.path.exists(args.ckpt_path), f"Error: checkpoint [{args.ckpt_path}] Not Found!"
    mmg_model = load_model_checkpoint(mmg_model, args.ckpt_path)
    teacher_model = load_model_checkpoint(teacher_model, args.ckpt_path)

    mmg_model.train()
    teacher_model.eval()
    
    # encoding 먼저 진행한 후에 크기 보고 resize할지말지 정해야할 듯
    #   ## sample shape
    # assert (args.height % 16 == 0) and (args.width % 16 == 0), "Error: image size [h,w] should be multiples of 16!"
    # ## latent noise shape
    # h, w = args.height // 8, args.width // 8
    # frames = model.temporal_length if args.frames < 0 else args.frames
    # channels = model.channels #4
    # ## saving folders
    # os.makedirs(args.savedir, exist_ok=True)
    
    #Logging folder
    name = Path(args.config).stem
    folder_name = "debug" if args.is_debug else name + datetime.datetime.now().strftime("-%Y-%m-%dT%H-%M-%S")
    output_dir = os.path.join(args.output_dir, folder_name)
    if args.is_debug and os.path.exists(output_dir):
        os.system(f"rm -rf {output_dir}")

    *_, config = inspect.getargvalues(inspect.currentframe())
    
     # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    run = wandb.init(project="MMG_videocrafter", name=folder_name, config=config)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/samples", exist_ok=True)
    os.makedirs(f"{output_dir}/sanity_check", exist_ok=True)
    os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
    
    #training_params = mmg_model.parameters()
    trainable_params = list(filter(lambda p: p.requires_grad, mmg_model.parameters()))

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    num_trainable_params = sum(p.numel() for p in trainable_params)
    print(f"Number of trainable parameters: {num_trainable_params}")
    
    num_all_params = sum(p.numel() for p in mmg_model.parameters())
    print(f"Number of mmg_model parameters: {num_all_params}")
    
    train_dataset = WebVid10M(
        csv_path="scripts/evaluation/data/audiocaps.csv",
        video_folder="scripts/evaluation/data/test_trimmed_audiocaps",
        sample_size=256,
        sample_stride=4, sample_n_frames=16,
        is_image=False,
    )    
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
    )
    
    # Scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    
    #total_batch_size = args.train_batch_size * args.num_processes * args.gradient_accumulation_steps
    
    logging.info("***** Running training *****")
    logging.info(f"  Num examples = {len(train_dataset)}")
    logging.info(f"  Num Epochs = {num_train_epochs}")
    logging.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    #logging.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logging.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logging.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps))
    progress_bar.set_description("Steps")
    
    
    for epoch in range(first_epoch, num_train_epochs):
        mmg_model.train()
        for step, batch in enumerate(train_dataloader):
            if args.cfg_random_null_text:
                batch['text'] = [name if random.random() > args.cfg_random_null_text_ratio else "" for name in batch['text']]
                
            # Data batch sanity check
            if step == 0:
                pixel_values, texts = batch['pixel_values'], batch['text']
                
            
                for idx, (pixel_value, text) in enumerate(zip(pixel_values, texts)):
                    pixel_value = pixel_value[None, ...]
                    save_videos_grid(pixel_value, f"{output_dir}/sanity_check/{'-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'{step}-{idx}'}.gif", rescale=True)
            
            # print("#######################################################################3")
            # print(batch["pixel_values"].shape)
            # print("#######################################################################3")
            # break
            
            ### >>>> Training >>>> ###
                
            # Convert videos to latent space            
            pixel_values = batch["pixel_values"].cuda(gpu_no)
            video_length = pixel_values.shape[1]
            with torch.no_grad():
                #pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
                latents = mmg_model.encode_first_stage_2DAE(pixel_values) #scale factor가 이미 곱해져있음

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            
            # Sample a random timestep for each video
            timesteps = torch.randint(0, 1000, (bsz,), device=latents.device)
            timesteps = timesteps.long()
            
            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents, GT_noise = mmg_model.q_sample(latents, timesteps)
            
            # conditioned_embeddings = []
            # with torch.no_grad():
            #     for text in batch["text"]:
            #         embedding = mmg_model.get_learned_conditioning(text)
            #         conditioned_embeddings.append(embedding)

            # # 배치 단위로 텍스트 임베딩을 합침
            # c_emb= torch.stack(conditioned_embeddings)
            with torch.no_grad():
                c_emb = mmg_model.get_learned_conditioning(batch["text"])
            
            print("conditioned_embeddings.shape: ", c_emb.shape)
            mmg_output = mmg_model.apply_model(noisy_latents, timesteps, c_emb)
            teacher_output = teacher_model.apply_model(noisy_latents, timesteps, c_emb)

            loss = F.mse_loss(mmg_output.float(), teacher_output.float(), reduction="mean")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mmg_model.parameters(), args.max_grad_norm)
            optimizer.step()
            
            lr_scheduler.step()
            progress_bar.update(1)
            global_step += 1
            
            ### <<<< Training <<<< ###
            
            # Wandb logging
            if args.is_main_process and (not args.is_debug) and args.use_wandb:
                wandb.log({"train_loss": loss.item()}, step=global_step)
                
            # Save checkpoint
            if args.is_main_process and (global_step % args.checkpointing_steps == 0 or step == len(train_dataloader) - 1):
                save_path = os.path.join(output_dir, f"checkpoints")
                state_dict = {
                    "epoch": args.epoch,
                    "global_step": global_step,
                    "state_dict": mmg_model.state_dict(),
                }
                if step == len(train_dataloader) - 1:
                    torch.save(state_dict, os.path.join(save_path, f"checkpoint-epoch-{epoch+1}.ckpt"))
                else:
                    torch.save(state_dict, os.path.join(save_path, f"checkpoint.ckpt"))
                logging.info(f"Saved state to {save_path} (global_step: {global_step})")
                
            
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            
            if global_step >= args.max_train_steps:
                break

if __name__ == '__main__':
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print("@CoLVDM Training: %s"%now)
    parser = get_parser()
    videocrafter_args = parser.parse_args()
    seed_everything(videocrafter_args.seed)
    rank, gpu_num = 0, 1
    train_mmg(videocrafter_args, gpu_num, rank)