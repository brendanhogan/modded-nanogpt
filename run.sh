#torchrun --standalone --nproc_per_node=8 train_gpt2.py
python3 -m torch.distributed.run --standalone --nproc_per_node=8 train_gpt2.py
