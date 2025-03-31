Here is the command I used to run the training on my 2x 8x Tesla setup:

```bash
torchrun \
    --nnodes=2 \
    --nproc_per_node=8 \
    --rdzv_id=unique_job_id_123 \
    --rdzv_backend=c10d \
    --rdzv_endpoint="node1:29500" \
    model.py \
     --checkpoint_dir /home/ubuntu/ddp-pt-transformer \
     --batch_size 8 \
     --learning_rate 0.0001 \
     --n_embd 512 \
     --num_heads 16 \
     --n_layers 16 \
     --block_size 350 \
     --train_iters 1000
```

I wasn't finished increasing the model size to use most of the GPU's memory.