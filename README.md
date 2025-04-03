# Hi!

This is a quick guide to help you train a transformer model on a remote machine.
In this repo I use my own implementation of the transformer I previously made following Karpathy's lecture.
I then adapted the code to use distributed data parallelism during training.
I describe the process to run this on a cloud computing service below.

## Cloud computing setup

I used Lambda Cloud to create a 2x 8x Tesla v100 setup because that was the cheapest aviailable on Lambda.
The advantage of Lambda over AWS or Paperspace is you don't have to request and wait (multiple business days) for a quota increase.
You can use other providers and configurations, though.
For different configurations you need to change the `nnodes` and `nproc_per_node` arguments.
Also, if your machines have less memory you may need to decrease the model size through the various parameters.

If you use Lambda, you create a file system and they automatically mount it in your home directory.
You will use its name, `<filesystem-name>` in the command below. If you use other providers, you will need to mount the file system yourself.

After starting your machines, add the IP and hostname mapping of all the child nodes on the `/etc/hosts` file of the main node.

## Distributed training command

Here is the command I used to run the training on my 2x 8x Tesla v100 setup:

```bash
torchrun \
    --nnodes=2 \
    --nproc_per_node=8 \
    --rdzv_id=unique_job_id_123 \
    --rdzv_backend=c10d \
    --rdzv_endpoint="node1:29500" \
    model.py \
     --checkpoint_dir /home/ubuntu/<filesystem-name> \
     --batch_size 8 \
     --learning_rate 0.0001 \
     --n_embd 512 \
     --num_heads 16 \
     --n_layers 16 \
     --block_size 350 \
     --train_iters 1000
```

## Notes

I wasn't finished increasing the model size to use most of the GPU's memory.
So this command isn't really optimized for the Tesla v100.
But the point of this project was just to train _something_ in parallel.
When I want to be as efficient as possible, I will also be using an implementation that includes flash attention and other optimizations.
