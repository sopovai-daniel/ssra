import torch
import torch.nn as nn
import torch.optim as optim
import math
from torch.utils.data import Dataset, DataLoader

# Nastavenie zariadenia (Apple Silicon GPU)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# --- 1. ARCHITEKTÚRA TVOJHO MODELU (DFKS JADRO) ---
class DFKS_Core(nn.Module):
    def __init__(self, vocab_size, d_model=64, threshold=2):
        super().__init__()
        self.d_model = d_model
        self.threshold = threshold
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.W_out = nn.Linear(d_model, vocab_size)

    def standard_attention(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_model)
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, v)

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len <= self.threshold:
            return self.standard_attention(x)
        if seq_len < 1:
            return torch.zeros(x.shape[0], 1, self.d_model).to(device)

        mid = seq_len // 2
        left_chunk = x[:, :mid, :]
        right_chunk = x[:, mid:, :]
        
        left_processed = self.forward(left_chunk)
        right_processed = self.forward(right_chunk)
        
        combined = torch.cat([left_processed, right_processed], dim=1)
        output = self.standard_attention(combined)
        output = output + (torch.randn_like(output) * 0.01) # Stochastická iskra
        
        return output

    def generate_next_token(self, x_tokens):
        x_embed = self.embedding(x_tokens)
        hidden_state = self.forward(x_embed)
        last_state = hidden_state[:, -1, :]
        logits = self.W_out(last_state)
        return torch.argmax(torch.softmax(logits, dim=-1), dim=-1).item()

# --- 2. DÁTOVÁ PIPELINE (DATASET) ---
class TextDataset(Dataset):
    def __init__(self, text_indices, seq_len):
        self.text_indices = text_indices
        self.seq_len = seq_len

    def __len__(self):
        return len(self.text_indices) - self.seq_len

    def __getitem__(self, idx):
        x = torch.tensor(self.text_indices[idx : idx + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.text_indices[idx + 1 : idx + self.seq_len + 1], dtype=torch.long)
        return x, y

# --- 3. POMOCNÁ FUNKCIA PRE TRÉNINGOVÉ LOGITY ---
def compute_sequence_logits(model, x_tokens):
    x_embed = model.embedding(x_tokens)
    hidden_states = model.forward(x_embed)
    logits = model.W_out(hidden_states)
    return logits

# --- 4. TRÉNINGOVÁ SĽUČKA ---
def train_dfks(model, dataset, epochs=5, batch_size=4, lr=5e-4):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    model.train()
    print(f"Spúšťam DFKS tréning na zariadení: {device}")
    print("----------------------------------------")

    for epoch in range(epochs):
        total_loss = 0
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            logits = compute_sequence_logits(model, x_batch)
            
            loss = criterion(logits.view(-1, logits.size(-1)), y_batch.view(-1))
            loss.backward()
            
            # Ochrana pred explóziou gradientov v rekurzii
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epocha [{epoch+1}/{epochs}] -> Priemerná strata (Loss): {avg_loss:.4f}")

# --- 5. SPUSTENIE SIMULÁCIE ---
if __name__ == "__main__":
    # Generovanie dummy slovníka (100 unikátnych slov) a textu (1000 tokenov)
    vocab_size = 100
    dummy_text = torch.randint(0, vocab_size, (1000,)).tolist()
    
    # Nastavenie dĺžky kontextu (8 slov)
    seq_len = 8
    dataset = TextDataset(dummy_text, seq_len)
    
    # Inicializácia modelu a presun na Apple Silicon GPU (MPS)
    model = DFKS_Core(vocab_size=vocab_size, d_model=64, threshold=2).to(device)
    
    # Spustenie tréningu
    train_dfks(model, dataset, epochs=5, batch_size=4, lr=5e-4)