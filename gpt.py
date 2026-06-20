import re
from pathlib import Path
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def strip_gutenberg(text):
    start = re.search(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", text, re.IGNORECASE)
    end = re.search(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", text, re.IGNORECASE)
    s = start.end() if start else 0
    e = end.start() if end else len(text)
    body = text[s:e].replace("\r\n", "\n").replace("\r", "\n")
    body = re.sub(r"^\s*Produced by[^\n]*\n", "", body, flags=re.IGNORECASE)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body.strip()


def load_data():
    if not Path("input.txt").exists():
        sources = [
            "https://raw.githubusercontent.com/GITenberg/The-Adventures-of-Sherlock-Holmes_1661/master/1661.txt",
            "https://raw.githubusercontent.com/GITenberg/The-Memoirs-of-Sherlock-Holmes_834/master/834.txt",
        ]
        parts = []
        for url in sources:
            raw = urllib.request.urlopen(url).read().decode("utf-8", errors="ignore")
            parts.append(strip_gutenberg(raw))
        combined = "\n\n".join(parts)
        combined = combined[:1115394]
        cut = combined.rfind("\n")
        if cut > len(combined) * 0.95:
            combined = combined[:cut]
        open("input.txt", "w", encoding="utf-8").write(combined)
    text = open("input.txt", "r", encoding="utf-8").read()
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
    return data, stoi, itos, len(chars)


class NextTokenDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y


class Head(nn.Module):
    def __init__(self, emb_dim, head_size, block_size, dropout=0.1):
        super().__init__()
        self.key = nn.Linear(emb_dim, head_size, bias=False)
        self.query = nn.Linear(emb_dim, head_size, bias=False)
        self.value = nn.Linear(emb_dim, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        T = x.shape[1]
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        wei = q @ k.transpose(-2, -1) * (k.size(-1) ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, num_heads, block_size, dropout=0.1):
        super().__init__()
        head_size = emb_dim // num_heads
        self.heads = nn.ModuleList([
            Head(emb_dim, head_size, block_size, dropout) for _ in range(num_heads)
        ])
        self.proj = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, emb_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.ReLU(),
            nn.Linear(4 * emb_dim, emb_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, emb_dim, num_heads, block_size, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(emb_dim)
        self.sa = MultiHeadAttention(emb_dim, num_heads, block_size, dropout)
        self.ln2 = nn.LayerNorm(emb_dim)
        self.ffwd = FeedForward(emb_dim, dropout)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, block_size,
                 emb_dim=128, num_heads=4, num_layers=4, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, emb_dim)
        self.position_embedding = nn.Embedding(block_size, emb_dim)
        self.blocks = nn.Sequential(*[
            Block(emb_dim, num_heads, block_size, dropout) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(emb_dim)
        self.lm_head = nn.Linear(emb_dim, vocab_size)

    def forward(self, x):
        T = x.shape[1]
        pos = torch.arange(T, device=x.device)
        tok = self.token_embedding(x)
        pos = self.position_embedding(pos)[None]
        h = tok + pos
        h = self.blocks(h)
        h = self.ln_f(h)
        logits = self.lm_head(h)
        return logits


def sequence_cross_entropy(logits, targets):
    return F.cross_entropy(logits.transpose(1, 2), targets)


def train_one_epoch(model, loader, optimizer, device, max_steps=None):
    model.train()
    total_loss, total_count = 0.0, 0
    for step, (xb, yb) in enumerate(loader):
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = sequence_cross_entropy(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        total_count += xb.size(0)
        if max_steps is not None and step + 1 >= max_steps:
            break
    return total_loss / total_count


@torch.no_grad()
def sample_gpt(model, block_size, stoi, itos, device, start_text, max_new_tokens):
    model.eval()
    context = torch.zeros((1, block_size), dtype=torch.long, device=device)
    for ch in start_text:
        if ch in stoi:
            ix = torch.tensor([[stoi[ch]]], device=device)
            context = torch.cat([context[:, 1:], ix], dim=1)
    out = list(start_text)
    for _ in range(max_new_tokens):
        logits = model(context)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        ix = torch.multinomial(probs, num_samples=1)
        out.append(itos[ix.item()])
        context = torch.cat([context[:, 1:], ix], dim=1)
    return "".join(out)


def main():
    block_size = 64
    data, stoi, itos, vocab_size = load_data()
    dataset = NextTokenDataset(data, block_size)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT(vocab_size, block_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for epoch in range(20):
        train_loss = train_one_epoch(model, loader, optimizer, device, max_steps=300)
        print(f"epoch {epoch:2d} | train loss {train_loss:.4f}")

    print(sample_gpt(model, block_size, stoi, itos, device,
                     start_text="Sherlock Holmes ", max_new_tokens=2000))


if __name__ == "__main__":
    main()
