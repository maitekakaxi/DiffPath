"""
Train a diffusion model on images.
"""
import sys
sys.path.append('/root/workspace/DiffPath/improved-diffusion')

import argparse
import json, torch, os
import numpy as np
from improved_diffusion import dist_util, logger
from improved_diffusion.image_datasets import load_data
from improved_diffusion.Path_datasets import load_data_path
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from transformers import AutoTokenizer
from improved_diffusion.train_util import TrainLoop
from transformers import set_seed
from functools import partial
from improved_diffusion.test_util import get_weights, compute_logp
from improved_diffusion.rounding import load_models
import torch.distributed as dist
import wandb

def main():
    args = create_argparser().parse_args()
    set_seed(args.seed) 
    dist_util.setup_dist()
    logger.configure()


    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev()) #  DEBUG **
    # model.cuda() #  DEBUG **

    pytorch_total_params = sum(p.numel() for p in model.parameters())

    logger.log(f'the parameter count is {pytorch_total_params}')
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log(f'saving the hyperparameters to {args.checkpoint_path}/training_args.json')
    with open(f'{args.checkpoint_path}/training_args.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    wandb.init(
        project=os.getenv("WANDB_PROJECT", "DiffPath"),
        name=args.checkpoint_path,
    )
    wandb.config.update(args.__dict__, allow_val_change=True)

    logger.log("creating data loader...")


    model22 = None

    data = load_data_path(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            class_cond=args.class_cond,
            data_args = args,
            task_mode=args.modality,
            padding_mode=args.padding_mode, 
            model=model22,
        )
    next(data)
    model2 = load_models(args.modality, args.experiment,  args.in_channel,
                                        args.checkpoint_path, extra_args=args)

    data_valid = load_data_path(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            class_cond=args.class_cond,
            data_args=args,
            task_mode=args.modality,
            padding_mode=args.padding_mode,  # block, pad
            split='valid',
            model=model2,
        )

    def get_mapping_func(args, diffusion, data):
        model2 = load_models(args.modality, args.experiment,  args.in_channel,
                                        args.checkpoint_path, extra_args=args)
        model3 = get_weights(model2, args)
        print(model3, model3.weight.requires_grad)
        mapping_func = partial(compute_logp, args, model3.cuda())
        diffusion.mapping_func = mapping_func
        return mapping_func

    get_mapping_func(args, diffusion, data)

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        checkpoint_path=args.checkpoint_path,
        gradient_clipping=args.gradient_clipping,
        eval_data=data_valid,
        eval_interval=args.eval_interval
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=50,
        save_interval=50000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        seed=101,
        gradient_clipping=-1.0,
        eval_interval=2000,
        checkpoint_path='/root/workspace/DiffPath/improved-diffusion/diffusion_models'
    )
    Path_defaults = dict(modality='Path',
                         experiment='random',
                         model_arch='conv-unet',
                         e2e_train='e2e_data',
                         padding_mode='block',
                         preprocessing_num_workers=1)
    defaults.update(model_and_diffusion_defaults())
    defaults.update(Path_defaults)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
