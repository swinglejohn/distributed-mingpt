import torch
import torch.nn as nn
import torch.nn.functional as F

# Distributed training
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group, destroy_process_group, is_initialized

import os


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


# Create a Dataset
class CharDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        # Subtract block_size because we need block_size + 1 elements for x and y
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # Grab chunk of (block_size + 1) characters
        chunk = self.data[idx : idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


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


def train(local_rank, global_rank):
    # Setup datasets and dataloaders
    train_dataset = CharDataset(train_data, block_size)
    val_dataset = CharDataset(val_data, block_size)

    # Use DistributedSampler
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    # Use DataLoader
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, sampler=val_sampler, pin_memory=True
    )

    model = LanguageModel()
    map_location = {"cuda:%d" % 0: "cuda:%d" % local_rank}
    if os.path.exists("latest_checkpoint.pth"):  # Load latest checkpoint
        print(f"Rank {global_rank}: Loading checkpoint...")
        model.load_state_dict(
            torch.load("latest_checkpoint.pth", map_location=map_location)
        )
    print(
        f"Rank {global_rank}: There are {sum(p.numel() for p in model.parameters())} parameters"
    )
    model.to(device)

    if is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])
        raw_model = model.module
    else:
        raw_model = model

    if global_rank == 0:
        start = torch.zeros((1, 1), dtype=torch.long, device=device)
        print("Initial generation:")
        print(decode(raw_model.generate(start, 50).view(-1)))

    adam = torch.optim.AdamW(raw_model.parameters(), lr=learning_rate)

    @torch.no_grad()
    def eval():
        raw_model.eval()
        out = {}
        val_losses = torch.zeros(eval_iters)
        val_iter = iter(val_loader)
        for i in range(eval_iters):
            try:
                bx, by = next(val_iter)
            except StopIteration:
                val_sampler.set_epoch(i)
                val_iter = iter(val_loader)
                bx, by = next(val_iter)
            bx, by = bx.to(device), by.to(device)
            _, loss = model(bx, by)
            val_losses[i] = loss.item()
        out["val"] = val_losses.mean()
        raw_model.train()
        return out

    train_iter = iter(train_loader)
    for i in range(train_iters + 1):
        if i % len(train_loader) == 0:
            train_sampler.set_epoch(i // len(train_loader))

        try:
            bx, by = next(train_iter)
        except StopIteration:
            train_sampler.set_epoch((i // len(train_loader)) + 1)
            train_iter = iter(train_loader)
            bx, by = next(train_iter)

        bx, by = bx.to(device), by.to(device)

        is_sync_step = (i % sync_interval == 0) or (i == train_iters)
        if is_initialized() and not is_sync_step:
            with model.no_sync():
                _, loss = model(bx, by)
                loss.backward()
        else:
            _, loss = model(bx, by)
            loss.backward()
            adam.step()
            adam.zero_grad()

        if i % eval_interval == 0 or i == train_iters:
            losses = eval()
            valloss = losses["val"]
            if global_rank == 0:
                print(
                    f"Iter: {i:>5}, Step Loss: {loss.item():.4f}, Validation Loss: {valloss:.4f}"
                )

            if global_rank == 0:
                print(f"Saving checkpoint at iter {i}...")
                torch.save(raw_model.state_dict(), "latest_checkpoint.pth")

    if global_rank == 0:
        print("Final generation:")
        start = torch.zeros((1, 1), dtype=torch.long, device=device)
        print(decode(raw_model.generate(start, 500).view(-1)))


if __name__ == "__main__":
    if "LOCAL_RANK" in os.environ and "RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ["RANK"])
        init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        print(f"Initializing DDP: Global Rank {global_rank}, Local Rank {local_rank}")
    else:
        local_rank = 0
        global_rank = 0
        print("Running in non-distributed mode.")

    train(local_rank, global_rank)

    if is_initialized():
        destroy_process_group()
