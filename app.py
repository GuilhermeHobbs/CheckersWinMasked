from flask import Flask, request
from flask_cors import CORS

import numpy as np
import torch
import torch.nn as nn
import sys

from torch.nn import functional as F

app = Flask(__name__)
CORS(app)  # This enables CORS for all routes

block_size = 100
vocab_size = 34
device = 'cuda' if torch.cuda.is_available() else 'cpu'

n_embd = 400 # 64
n_head = 16  # 4
n_layer = 12 # 8
dropout = 0.2
# ------------

@torch.no_grad()
class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss


model = BigramLanguageModel()
m = model.to(device)

m.load_state_dict(torch.load('winMINI(1).pth', map_location=torch.device('cpu')))

model.eval()  # Disable dropout

context = torch.Tensor([[0]]).int().to(device)  

print("antess")

@app.route('/move')
def ask_name():
    global context
    global m
    
    a = request.args.get('a', '')  
    b = request.args.get('b', '')
    c = request.args.get('c', '')
    
    a = int(a)
    if a==0:
        context = torch.Tensor([[0]]).int().to(device)
        return ""
        
    b = int(b)   
 
    context = torch.cat([context, torch.Tensor([[a,b]]).to(device)], dim=1)
    if c != '':
        c = int(c)
        context = torch.cat([context, torch.Tensor([[c]]).to(device)], dim=1)
    
    logits, _ = m(context.int())
    logits = logits[-1,-1] 
    d = logits.argmax()
    
    context = torch.cat([context, torch.Tensor([[d]]).to(device)], dim=1)

    logits, _ = m(context.int())
    logits = logits[-1,-1] 
    e = logits.argmax()

    context = torch.cat([context, torch.Tensor([[e]]).to(device)], dim=1)

    if e==33:
        logits, _ = m(context.int())
        logits = logits[-1,-1] 
        f = logits.argmax()
        context = torch.cat([context, torch.Tensor([[f]]).to(device)], dim=1)

        logits, _ = m(context.int())
        logits = logits[-1,-1] 
        g = logits.argmax()
        context = torch.cat([context, torch.Tensor([[g]]).to(device)], dim=1)
            
        if g==33:
            logits, _ = m(context.int())
            logits = logits[-1,-1] 
            h = logits.argmax()
            context = torch.cat([context, torch.Tensor([[h]]).to(device)], dim=1)        

    
    print("Cooontext:",context, flush=True)  # Force immediate flushing
    sys.stdout.flush()
    
    if e==33 and g==33:
        return str(d.item())+"-33-"+str(f.item())+"-33-"+str(h.item())
    elif e==33:
        return str(d.item())+"-33-"+str(f.item())
    else:
        return str(d.item())+"-"+str(e.item())


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)  # Render requires explicit host/port
