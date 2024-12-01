"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import sys
sys.path.append('/root/workspace/DiffPath/improved-diffusion')

import argparse
import os, json

import numpy as np
import torch as th
import torch.distributed as dist
from transformers import set_seed
from improved_diffusion.rounding import rounding_func, load_models

from improved_diffusion.test_util import get_weights, denoised_fn_round

from improved_diffusion import dist_util, logger
from functools import partial
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

def save_tensor(tensor, file_path):
    np.savetxt(file_path, tensor.cpu().numpy(), fmt='%i', delimiter=' ')
def main():
    set_seed(102)
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    config_path = os.path.join(os.path.split(args.model_path)[0], "training_args.json")
    print(config_path)
    with open(config_path, 'rb', ) as f:
        training_args = json.load(f)
    training_args['batch_size'] = args.batch_size
    args.__dict__.update(training_args)
    args.sigma_small = True


    if args.experiment == 'random1': args.experiment = 'random'
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.log(f'the parameter count is {pytorch_total_params}')

    print(diffusion.rescale_timesteps, 'a marker for whether we are in the debug mode')
    model.to(dist_util.dev())
    model.eval() # DEBUG



    if args.experiment_mode == 'conditional_gen':
        from improved_diffusion.Path_datasets import load_data_path
        model2, tokenizer = load_models(args.modality, args.experiment, args.model_name_or_path, args.in_channel,
                                        os.path.split(args.model_path)[0])
        print('conditional generation mode --> load data')

        data = load_data_path(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            class_cond=args.class_cond,
            data_args=args,
            model=model2,
            deterministic=True,
            task_mode=args.modality,
            padding_mode=args.padding_mode,  
            split=args.split,
        )

    model2= load_models(args.modality, args.experiment, args.model_name_or_path, args.in_channel,
                                    os.path.split(args.model_path)[0])
    if args.training_mode.startswith('e2e'):
        print('e2e, load the right model embeddings', '*'*80)
        model2.weight = th.nn.Parameter(model.word_embedding.weight.clone().cpu())

    logger.log("sampling...")
    all_images = []
    all_labels = []
    print(args.num_samples)
    model3 = get_weights(model2, args)
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.experiment_mode == 'conditional_gen':
            batch, model_kwargs = next(data)
            model_kwargs.pop('input_ids')
            if args.mbr_sample > 1:
                model_kwargs = {k: v.to(dist_util.dev()).repeat_interleave(args.mbr_sample, dim=0) for k, v in model_kwargs.items()}
            else:
                model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}
            print([(k, v.shape) for (k,v) in model_kwargs.items()])
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        if args.model_arch == '1d-unet':
            if args.mbr_sample > 1 and args.experiment_mode == 'conditional_gen':
                sample_shape = (args.batch_size * args.mbr_sample, args.in_channel, args.image_size ** 2)
            else:
                sample_shape = (args.batch_size,  args.in_channel, args.image_size ** 2)
        else:
            if args.mbr_sample > 1 and args.experiment_mode == 'conditional_gen':
                sample_shape = (args.batch_size * args.mbr_sample, args.image_size ** 2, args.in_channel)
            else:
                sample_shape = (args.batch_size, 144, args.in_channel)
        print(sample_shape)
        sample = sample_fn(
            model,
            sample_shape,
            noise=None,
            clip_denoised=args.clip_denoised,
            #denoised_fn=None,
            denoised_fn=partial(denoised_fn_round, args, model3.cuda()) if args.clamp == 'clamp' else None,
            model_kwargs=model_kwargs,
            top_p =args.top_p,
        )

        if args.model_arch == '1d-unet':
            print(sample.shape)
            sample = sample.permute(0, 2, 1)
        print(sample.shape)
        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample) 
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    print(arr.shape, 'full shape')
    arr = arr[: args.num_samples * args.mbr_sample]

    if diffusion.training_mode.startswith('e2e'):
        print('decoding for e2e', )
        print(arr.shape)
        x_t = th.tensor(arr).cuda()
        if args.model_arch == 'conv-unet':
            reshaped_x_t = x_t.view(x_t.size(0), -1, x_t.size(-1))
        else:
            reshaped_x_t = x_t
        logits = model.get_logits(reshaped_x_t)  # bsz, seqlen, vocab
        cands = th.topk(logits, k=1, dim=-1)
        sample = cands.indices
        sample_squeezed = sample.squeeze(-1)
        sample_np = sample_squeezed.cpu().numpy()
        np.savetxt('/root/workspace/Diffusion-LM/improved-diffusion/generation_outputs/path_integer_sequences_chengdu_100000.txt', sample_np, fmt='%i', delimiter=' ')
    dist.barrier()
    logger.log("sampling complete")



def create_argparser():
    defaults = dict(
        clip_denoised=False,
        num_samples=1000,#10000,
        batch_size=1000,
        use_ddim=False,
        mbr_sample=1,
        model_path="/root/workspace/DiffPath/improved-diffusion/diffusion_models/diff_path_chengdu_block_rand128_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd102_xstart_e2e/ema_0.9999_100000.pt",
        model_arch='conv-unet',
        verbose='yes',
        out_dir="/root/workspace/DiffPath/improved-diffusion/generation_outputs"
    )
    Path_defaults = dict(modality='Path',
                         experiment='random',
                         model_arch='conv-unet',
                         e2e_train='e2e_data',
                         padding_mode='block',
                         preprocessing_num_workers=1,
                         emb_scale_factor=1.0, top_p=-1., split='valid', clamp='clamp')
    defaults.update(model_and_diffusion_defaults())
    defaults.update(Path_defaults)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
