# from PIL import Image
# import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
import sys, os
import torch
import pandas as pd
import sys
sys.path.append('/root/workspace/DiffPath/improved-diffusion')

class PathDataset(Dataset):
 def __init__(self, data, model_emb):
  self.data = data
  self.model_emb = model_emb

 def __len__(self):
  return len(self.data)

 def __getitem__(self, idx):
  sequence = self.data.iloc[idx].values # 获取序列
  input_ids = torch.tensor(sequence, dtype=torch.long) # 转换为PyTorch张量
  embedded_sequence = self.model_emb(input_ids) # 通过embedding层处理序列
  out_dict = {}
  out_dict['input_ids'] = np.array(input_ids)
  numpy_sequence = embedded_sequence.detach().numpy()
  reshaped_sequence = np.transpose(numpy_sequence, (0, 1)) # 重新排列维度
  return reshaped_sequence,out_dict
def load_data_path(
    *, data_dir, batch_size, image_size, class_cond=False, deterministic=False, data_args=None, 
        task_mode='roc', model=None, padding_mode='block', split='train',
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """
    print('hello loading text data. ')

    if data_args.experiment.startswith('random') and model is None:
        model = None
        print('new data (path data),loading initialized random embeddings. ')
    elif data_args.experiment.startswith('random') and model is not None:
        print('loading initialized random embeddings. ')


    if task_mode == 'path_xian':
        model = torch.nn.Embedding(data_args.vocab_size, data_args.in_channel,padding_idx=2675)
        print('initializing the random embeddings', model)
        torch.nn.init.normal_(model.weight)
        path_save =f'{data_args.checkpoint_path}/random_emb.torch'
        print(f'save the random encoder to {data_args.checkpoint_path}/random_emb.torch')
        torch.save(model.state_dict(), path_save)
        
        train_data = pd.read_csv('.../')
        val_data = pd.read_csv('.../')
        
        if split =='train':
            dataset = PathDataset(train_data,model_emb = model)
        else:
            dataset = PathDataset(val_data,model_emb = model)
    elif task_mode == 'path_chengdu':
        model = torch.nn.Embedding(data_args.vocab_size, data_args.in_channel,padding_idx=2865)
        print('initializing the random embeddings', model)
        torch.nn.init.normal_(model.weight)
        
        path_save =f'{data_args.checkpoint_path}/random_emb.torch'
        #path_save =f'/root/workspace/Diffusion-LM/improved-diffusion/diffusion_models/diff_path_chengdu_block_rand128_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd102_xstart_e2e/random_emb.torch'
        print(f'save the random encoder to {data_args.checkpoint_path}/random_emb.torch')
        torch.save(model.state_dict(), path_save)
        
        train_data = pd.read_csv('.../')
        val_data = pd.read_csv('.../')
        
        if split =='train':
            dataset = PathDataset(train_data,model_emb = model)
        else:
            dataset = PathDataset(val_data,model_emb = model)
#
    if deterministic:

        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,  # 20,
            drop_last=True,
            shuffle=False,
            num_workers=1,
        )

    else:
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,  # 20,
            drop_last=True,
            shuffle=True,
            num_workers=1,
        )
    while True:
        yield from data_loader