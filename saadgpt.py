import os
import math
import time
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from transformers import AutoTokenizer

# ==========================================
# 0. Setup Logging with Timestamps
# ==========================================
logging.basicConfig(
    format="[%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger("GPT_Termux_Logger")

logger.info("=" * 50)
logger.info("INITIALIZING CUSTOM GPT TRAINING PIPELINE ON PHONE")
logger.info("=" * 50)

# ==========================================
# 1. Multithreading Configuration for S21
# ==========================================
NUM_THREADS = 4
torch.set_num_threads(NUM_THREADS)
torch.set_num_interop_threads(2)
os.environ["OMP_NUM_THREADS"] = str(NUM_THREADS)
os.environ["MKL_NUM_THREADS"] = str(NUM_THREADS)

logger.info(f"Configured PyTorch CPU execution threads: {torch.get_num_threads()}")
logger.info(f"Configured Inter-op threads: {torch.get_num_interop_threads()}")

device = torch.device("cpu")

# ==========================================
# 2. Custom GPT Architecture with Logs
# ==========================================

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, max_len, dropout=0.1):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.head_dim = d_model // n_head

        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.register_buffer("bias", torch.tril(torch.ones(max_len, max_len)).view(1, 1, max_len, max_len))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = torch.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class FeedForward(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, max_len, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, max_len, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = FeedForward(d_model, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class CustomGPT(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_head=4, n_layer=4, max_len=128, dropout=0.1):
        super().__init__()
        self.max_len = max_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_head, max_len, dropout) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.tok_emb.weight = self.head.weight

    def forward(self, idx):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)

        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits

# ==========================================
# 3. Data Processing & Dataset with Logs
# ==========================================

class CausalTranslationDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_len=128):
        logger.info(f"Reading dataset CSV from: {csv_file}")
        t0 = time.time()
        
        df = pd.read_csv(csv_file).dropna()
        logger.info(f"Loaded {len(df)} non-empty rows in {time.time() - t0:.3f}s")
        
        self.samples = []
        logger.info("Tokenizing and formatting sequences into 'English <SEP> Urdu'...")
        
        for idx, row in df.iterrows():
            text = f"{row['English']} {tokenizer.sep_token} {row['Urdu']}"
            encoded = tokenizer.encode(text, truncation=True, max_length=max_len)
            
            if len(encoded) < max_len:
                encoded = encoded + [tokenizer.pad_token_id] * (max_len - len(encoded))
            
            self.samples.append(torch.tensor(encoded[:max_len], dtype=torch.long))

        logger.info(f"Successfully processed {len(self.samples)} samples. Max context length: {max_len}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# ==========================================
# 4. Pipeline Execution
# ==========================================

CSV_PATH = "test.csv"
if not os.path.exists(CSV_PATH):
    logger.error(f"File {CSV_PATH} not found! Create the CSV before running.")
    raise FileNotFoundError(f"Missing {CSV_PATH}")

logger.info("Loading tokenizer from 'Helsinki-NLP/opus-mt-en-ur'...")
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ur")
if tokenizer.sep_token is None:
    tokenizer.add_special_tokens({'sep_token': '<SEP>'})
logger.info(f"Tokenizer loaded in {time.time() - t0:.2f}s | Vocab size: {len(tokenizer)}")

dataset = CausalTranslationDataset(CSV_PATH, tokenizer, max_len=128)
logger.info("Initializing DataLoader with 2 background workers...")
train_loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2, persistent_workers=True)

logger.info("Instantiating Custom GPT Model...")
model = CustomGPT(
    vocab_size=len(tokenizer),
    d_model=128,
    n_head=4,
    n_layer=4,
    max_len=128,
    dropout=0.2
).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info(f"Model initialized. Total Parameters: {total_params:,} | Trainable: {trainable_params:,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

EPOCHS = 10
logger.info("-" * 50)
logger.info(f"STARTING TRAINING LOOP FOR {EPOCHS} EPOCHS")
logger.info("-" * 50)

total_start_time = time.time()

for epoch in range(EPOCHS):
    epoch_start_time = time.time()
    model.train()
    total_loss = 0.0
    step_times = []

    logger.info(f"--- Epoch {epoch + 1}/{EPOCHS} Started ---")

    for step, batch in enumerate(train_loader):
        step_start = time.time()
        
        batch = batch.to(device)
        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        optimizer.zero_grad()
        logits = model(inputs)

        loss = criterion(logits.reshape(-1, len(tokenizer)), targets.reshape(-1))
        loss.backward()
        optimizer.step()

        step_elapsed = time.time() - step_start
        step_times.append(step_elapsed)
        total_loss += loss.item()

        # Log detailed info every 5 steps
        if (step + 1) % 5 == 0 or (step + 1) == len(train_loader):
            logger.info(
                f"Epoch [{epoch + 1}/{EPOCHS}] | Step [{step + 1}/{len(train_loader)}] | "
                f"Batch Loss: {loss.item():.4f} | Step Time: {step_elapsed * 1000:.1f}ms"
            )

    epoch_elapsed = time.time() - epoch_start_time
    avg_loss = total_loss / len(train_loader)
    avg_step_time = (sum(step_times) / len(step_times)) * 1000

    logger.info(
        f"==> Epoch {epoch + 1} Complete | Avg Loss: {avg_loss:.4f} | "
        f"Epoch Time: {epoch_elapsed:.2f}s | Avg Step Time: {avg_step_time:.1f}ms"
    )
    logger.info("-" * 50)

total_elapsed = time.time() - total_start_time
logger.info(f"TRAINING COMPLETE IN {total_elapsed:.2f} SECONDS!")

# Save Checkpoint
SAVE_PATH = "custom_gpt_phone.pt"
logger.info(f"Saving model state dict to: {SAVE_PATH}")
torch.save(model.state_dict(), SAVE_PATH)
logger.info("Model saved successfully. Pipeline finished cleanly!")
