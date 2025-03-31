import torch
import torch.nn as nn
import torch.nn.functional as F

# Distributed training
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group, destroy_process_group

import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split


# Constants
batch_size = 32
block_size = 26  # length of the Time dimension or context
learning_rate = 1e-3
n_embd = 24
num_heads = 4  # head_size is n_embed // num_heads
n_layers = 6  # how many layers of multi headed attention + feedforward networks to use
drop_rate = 0.2  # dropout rate
eval_iters = 200
eval_interval = 500
sync_interval = 10
train_iters = 6000
device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(1337)

print()
print(f"Cuda available: {torch.cuda.is_available()}")
print(f"Using device: {device}")
print(torch.__version__)
print(f"Batch Size: {batch_size}")
print()


# Dummy variables to make Pylance happy :D
train_dataset = None
local_rank = -1
global_rank = -1
num_epochs = 100
step_number = 0
last_step = False

with open("shakespeare.txt", "r") as f:
    text = "".join(f.readlines())
vocab = sorted(set(text))

itos = {i: c for i, c in enumerate(vocab)}
stoi = {c: i for i, c in enumerate(vocab)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda nums: "".join(itos[n] for n in nums.tolist())

data = torch.tensor(encode(text), dtype=torch.long)
n = int(len(data) * 0.9)
train_data = data[:n]
val_data = data[n:]


def get_batch(split="train"):
    data = train_data if split == "train" else val_data
    ixs = torch.randint(0, len(data) - block_size, (batch_size,))
    # for i, ix in enumerate(ixs):
    bx = torch.stack([data[ix : ix + block_size] for ix in ixs]).to(device)
    by = torch.stack([data[ix + 1 : ix + block_size + 1] for ix in ixs]).to(device)
    return bx, by


# single head of attention
class SingleHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.keys = nn.Linear(n_embd, head_size, bias=False)
        self.queries = nn.Linear(n_embd, head_size, bias=False)
        self.values = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.drop = nn.Dropout(drop_rate)

    def __call__(self, x):
        # x should be (B=batch_size, T=block_size, C=n_embd)
        B, T, C = x.shape
        k = self.keys(x)  # k should be (B, T, head_size)
        q = self.queries(x)
        v = self.values(x)

        wei = q @ k.transpose(-1, -2) / (k.shape[-1] ** 0.5)  # yields (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = wei.softmax(dim=-1)
        wei = self.drop(wei)

        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList(
            SingleHead(n_embd // num_heads) for _ in range(num_heads)
        )
        self.proj = nn.Linear(n_embd, n_embd)
        self.drop = nn.Dropout(drop_rate)

    def __call__(self, x):
        x = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.drop(self.proj(x))


class SelfAttentionHeads(nn.Module):
    def __init__(self):
        super().__init__()
        self.keys = nn.Linear(n_embd, n_embd, bias=False)
        self.queries = nn.Linear(n_embd, n_embd, bias=False)
        self.values = nn.Linear(n_embd, n_embd, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.drop = nn.Dropout(drop_rate)
        self.proj = nn.Linear(n_embd, n_embd)
        self.drop2 = nn.Dropout(drop_rate)

    def __call__(self, x):
        # x should be (B=batch_size, T=block_size, C=n_embd)
        head_size = n_embd // num_heads
        B, T, C = x.shape
        k = self.keys(x)  # k should be (B, T, n_embd)
        k = k.view(B, T, num_heads, head_size)  # now k is (B, T, num_heads, head_size)
        q = self.queries(x).view(B, T, num_heads, head_size)
        v = self.values(x).view(B, T, num_heads, head_size)
        k = k.transpose(1, 2)  # now k is (B, num_heads, T, head_size)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        wei = q @ k.transpose(-1, -2) / (k.shape[-1] ** 0.5)  # yields (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = wei.softmax(dim=-1)
        wei = self.drop(wei)

        out = wei @ v  # (B, num_heads, T, head_size)
        out = out.transpose(1, 2)  # undo the earlier transpose
        out = out.reshape(B, T, C)
        return self.drop2(self.proj(out))


class FeedFoward(nn.Module):
    def __init__(self):
        super().__init__()
        self.nets = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.ReLU(),
            nn.Linear(n_embd * 4, n_embd),  # projection back into the residual stream
            nn.Dropout(drop_rate),
        )

    def __call__(self, x):
        return self.nets(x)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.sahs = SelfAttentionHeads()
        self.ff = FeedFoward()
        self.l_norm1 = nn.LayerNorm(n_embd)
        self.l_norm2 = nn.LayerNorm(n_embd)

    def __call__(self, x):
        x = x + self.sahs(self.l_norm1(x))
        x = x + self.ff(self.l_norm2(x))
        return x


class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.C = nn.Embedding(len(vocab), n_embd)
        self.pos_table = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(*[Block() for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm

        self.lm_head = nn.Linear(n_embd, len(vocab))

    def __call__(self, idx, targets=None):
        B, T = idx.shape
        emb = self.C(
            idx
        )  # (B, T, C)  # where T can be less than block size (e.g. during generation)
        pos = self.pos_table(torch.arange(T, device=device))
        x = emb + pos
        x = self.blocks(x)
        x = self.ln_f(x)

        out = self.lm_head(x)
        B, T, C = out.shape

        loss = None
        if targets is not None:
            out = out.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(out, targets)

        return out, loss

    def generate(self, x, n):
        for _ in range(n):
            logits, _ = self(x[:, -block_size:])
            logits = logits[:, -1, :]  # get the last timestep
            probs = logits.softmax(1)
            y = torch.multinomial(probs, 1)
            x = torch.cat((x, y), 1)
        return x


def train():
    model = LanguageModel()
    if os.path.exists("latest_checkpoint.pth"):  # Load latest checkpoint
        # Also load optimizer state and other variables needed to restore the training state
        model.load_state_dict(torch.load("latest_checkpoint.pth"))
    print(f"There are {sum(p.numel() for p in model.parameters())} parameters")
    model.to(device)

    model = DistributedDataParallel(model, device_ids=[local_rank])

    start = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(start, 50).view(-1)))

    adam = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    @torch.no_grad()
    def eval():
        model.eval()
        out = {}
        for data_type in ["train", "val"]:
            losses = torch.zeros(eval_iters)
            for i in range(eval_iters):
                bx, by = get_batch(data_type)
                _, loss = model(bx, by)
                losses[i] = loss.item()
                out[data_type] = losses.mean()
        model.train()
        return out

    # training loop
    for i in range(train_iters + 1):
        if i % sync_interval == 0:
            bx, by = get_batch()
            _, loss = model(bx, by)  # forward
            loss.backward()  # Backward step + gradient SYNCHRONIZATION
            adam.step()
            model.zero_grad()
        else:
            with model.no_sync():
                bx, by = get_batch()
                _, loss = model(bx, by)  # forward
                loss.backward()  # Backward step + gradient ACCUMULATION

        if i % eval_interval == 0:
            losses = eval()
            valloss, traloss = losses["val"], losses["train"]
            print(
                f"Global Rank: {global_rank}, Iter: {i:>4}, Training Loss: {traloss:.4f}, Validation Loss: {valloss:.4f}"
            )

        if global_rank == 0:  # Only save on rank 0
            # Also save the optimizer state and other variables needed to restore the training state
            torch.save(model.state_dict(), "latest_checkpoint.pth")

    print(decode(model.generate(start, 500).view(-1)))


if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])

    init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)  # Set the device to local rank

    train()

    destroy_process_group()
