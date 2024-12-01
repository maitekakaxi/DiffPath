
-----------------------------------------------------
## Conda Setup:
```python 
conda install mpi4py
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install -e improved-diffusion/ 
pip install -e transformers/
pip install spacy==3.2.4
pip install datasets==1.8.0 
pip install huggingface_hub==0.4.0 
pip install wandb
```

-----------------------------------------------------
## Train:

```cd improved-diffusion; mkdir diffusion_models;```

```python scripts/run_train.py --diff_steps 2000 --model_arch transformer --lr 0.0001 --lr_anneal_steps 350000 --seed 101 --noise_schedule sqrt --in_channel 128 --modality path_chengdu --padding_mode block --app "--predict_xstart True --training_mode e2e --vocab_size 2866  --e2e_train ../datasets/e2e_data " --notes xstart_e2e```

-------------------
