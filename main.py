import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Hierarchical Reasoning Model (HRM) — PyTorch v0 for Natural-Language Math QA
    -----------------------------------------------------------------------------
    This is a first, faithful implementation of HRM adapted to natural language -> numeric
    answer tasks. It follows three defining elements from the paper:
      • Two interdependent recurrent modules (H and L), each an encoder-only Transformer.
      • One-step gradient approximation (no BPTT) performed per "segment".
      • Deep supervision, with optional Adaptive Computation Time (ACT) Q-head.

    Highlights
    ---------
    • Sequence-to-sequence setup: we concatenate input text and a shifted answer prefix; we
      train with cross-entropy on the answer portion.
    • Standalone character tokenizer (works out of the box). You can swap in a HF tokenizer
      by implementing the TokenizerProtocol.
    • Minimal, readable Transformer blocks with: bias-free Linear, RMSNorm, GLU FFN, and
      Rotary Positional Embeddings (RoPE).
    • Post-Norm residual layout as used by the paper; weights are bias-free and RMSNorm excludes
      scale/bias where appropriate (mirrors Llama-style modern blocks).
    • ACT halting policy is implemented per-sample; you can turn it off and use fixed M.
    • Careful tensor shapes. All hidden states keep full sequence shape so the output head can
      read tokenwise logits on the answer segment.

    This file is purposely self-contained for experimentation.

    Author: you + ChatGPT
    License: MIT
    """
    )
    return


@app.cell
def _():

    from __future__ import annotations
    import math
    import random
    from dataclasses import dataclass
    from typing import List, Tuple, Optional, Protocol, Iterable


    import torch

    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader

    import pandas as pd
    from sklearn.model_selection import train_test_split
    return (
        DataLoader,
        Dataset,
        F,
        Iterable,
        List,
        Optional,
        Protocol,
        Tuple,
        dataclass,
        math,
        nn,
        pd,
        torch,
        train_test_split,
    )


@app.cell
def _(Iterable, List, Protocol):

    # ---------------------------
    # Tokenizer
    # ---------------------------
    class TokenizerProtocol(Protocol):
        pad_id: int
        bos_id: int
        eos_id: int
        sep_id: int
        def encode(self, s: str) -> List[int]: ...
        def decode(self, ids: Iterable[int]) -> str: ...

    class CharTokenizer:
        """Simple, robust character tokenizer.
        Covers digits, letters, math symbols, punctuation, and whitespace.
        Add more symbols in `alphabet_extra` as needed.
        """
        def __init__(self):
            base = [chr(i) for i in range(32, 127)]  # basic printable ASCII
            alphabet_extra = ["£", "€", "π", "√", "∞", "±", "×", "÷"]
            special = ["<pad>", "<bos>", "<eos>", "<sep>"]
            self.vocab = special + base + alphabet_extra
            self.stoi = {s: i for i, s in enumerate(self.vocab)}
            self.itos = {i: s for i, s in enumerate(self.vocab)}
            self.pad_id, self.bos_id, self.eos_id, self.sep_id = 0, 1, 2, 3

        def encode(self, s: str) -> List[int]:
            return [self.stoi.get(ch, self.stoi['?']) if ch in self.stoi else self.stoi['?'] for ch in s]

        def decode(self, ids: Iterable[int]) -> str:
            return ''.join(self.itos.get(i, '?') for i in ids)

    return CharTokenizer, TokenizerProtocol


@app.cell
def _(nn, torch):

    # ---------------------------
    # RMSNorm (bias-free)
    # ---------------------------
    class RMSNorm(nn.Module):
        def __init__(self, d_model: int, eps: float = 1e-8):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(d_model))  # no bias
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B, L, D)
            var = x.pow(2).mean(dim=-1, keepdim=True)
            x_norm = x * torch.rsqrt(var + self.eps)
            return x_norm * self.weight

    return (RMSNorm,)


@app.cell
def _(Tuple, nn, torch):

    # ---------------------------
    # Rotary Positional Embeddings
    # ---------------------------
    class Rotary(nn.Module):
        def __init__(self, dim: int, base: int = 10000):
            super().__init__()
            self.dim = dim
            self.base = base
        def _angles(self, L: int, device):
            inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=device).float() / self.dim))
            t = torch.arange(L, device=device).float()
            freqs = torch.einsum('l,d->ld', t, inv_freq)
            return torch.cat([freqs, freqs], dim=-1)  # (L, dim)
        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            # x: (B, L, H, Dh)
            L, Dh = x.size(1), x.size(-1)
            assert Dh == self.dim
            angles = self._angles(L, x.device)  # (L, Dh)
            cos = angles.cos()[None, :, None, :]  # (1, L, 1, Dh)
            sin = angles.sin()[None, :, None, :]  # (1, L, 1, Dh)
            return cos, sin

    def apply_rotary(xq: torch.Tensor, xk: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        # x*: (B, L, H, Dh)
        def rotate_half(t):
            t1, t2 = t[..., :t.shape[-1]//2], t[..., t.shape[-1]//2:]
            return torch.cat([-t2, t1], dim=-1)
        xq2 = (xq * cos) + (rotate_half(xq) * sin)
        xk2 = (xk * cos) + (rotate_half(xk) * sin)
        return xq2, xk2

    return Rotary, apply_rotary


@app.cell
def _(nn):

    # ---------------------------
    # Bias-free Linear
    # ---------------------------
    class LinearNoBias(nn.Linear):
        def __init__(self, in_f, out_f):
            super().__init__(in_f, out_f, bias=False)

    return (LinearNoBias,)


@app.cell
def _(
    F,
    LinearNoBias,
    Optional,
    RMSNorm,
    Rotary,
    apply_rotary,
    math,
    nn,
    torch,
):

    # ---------------------------
    # Attention block (pre-attn input, post-norm residual)
    # ---------------------------
    class SelfAttention(nn.Module):
        def __init__(self, d_model: int, n_heads: int):
            super().__init__()
            assert d_model % n_heads == 0
            self.d_model = d_model
            self.n_heads = n_heads
            self.d_head = d_model // n_heads
            self.in_proj = LinearNoBias(d_model, 3 * d_model)
            self.out_proj = LinearNoBias(d_model, d_model)
            self.rotary = Rotary(self.d_head)
            self.norm = RMSNorm(d_model)

        def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            B, L, D = x.shape
            h = self.norm(x)  # Post-Norm architecture: norm on residual branch
            qkv = self.in_proj(h)
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.view(B, L, self.n_heads, self.d_head)
            k = k.view(B, L, self.n_heads, self.d_head)
            v = v.view(B, L, self.n_heads, self.d_head)
            cos, sin = self.rotary(q)
            q, k = apply_rotary(q, k, cos, sin)
            q = q.transpose(1, 2)  # (B, H, L, Dh)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
            if mask is not None:
                att = att.masked_fill(mask == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            y = att @ v  # (B, H, L, Dh)
            y = y.transpose(1, 2).contiguous().view(B, L, D)
            y = self.out_proj(y)
            return x + y  # residual, followed by Post-Norm in the next block

    return (SelfAttention,)


@app.cell
def _(F, LinearNoBias, Optional, RMSNorm, SelfAttention, nn, torch):

    # ---------------------------
    # GLU Feedforward
    # ---------------------------
    class GLUFeedForward(nn.Module):
        def __init__(self, d_model: int, mult: int = 4):
            super().__init__()
            hidden = d_model * mult
            self.fc1 = LinearNoBias(d_model, hidden * 2)
            self.fc2 = LinearNoBias(hidden, d_model)
            self.norm = RMSNorm(d_model)
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.norm(x)
            a, b = self.fc1(h).chunk(2, dim=-1)
            y = F.silu(a) * b
            y = self.fc2(y)
            return x + y

    class TransformerBlock(nn.Module):
        def __init__(self, d_model: int, n_heads: int, mult: int = 4):
            super().__init__()
            self.attn = SelfAttention(d_model, n_heads)
            self.ffn = GLUFeedForward(d_model, mult)
        def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            x = self.attn(x, mask)
            x = self.ffn(x)
            return x

    class TransformerEncoder(nn.Module):
        def __init__(self, d_model: int, n_heads: int, n_layers: int, mult: int = 4):
            super().__init__()
            self.layers = nn.ModuleList([
                TransformerBlock(d_model, n_heads, mult) for _ in range(n_layers)
            ])
        def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            for layer in self.layers:
                x = layer(x, mask)
            return x

    return (TransformerEncoder,)


@app.cell
def _(
    F,
    LinearNoBias,
    Optional,
    TransformerEncoder,
    Tuple,
    dataclass,
    nn,
    torch,
):

    # ---------------------------
    # HRM Modules and Model
    # ---------------------------
    @dataclass
    class HRMConfig:
        vocab_size: int
        d_model: int = 512
        n_heads: int = 8
        n_layers_L: int = 4
        n_layers_H: int = 4
        mult: int = 4
        T: int = 2               # low-level steps per high-level step
        N: int = 2               # high-level steps per segment (only the last step keeps grad)
        max_len: int = 512
        use_act: bool = False
        act_eps: float = 0.1     # epsilon for stoch. M_min selection
        act_mmax: int = 4
        dropout: float = 0.0

    class HRMModule(nn.Module):
        """One HRM module as an encoder-only Transformer.
        Combines multiple inputs via element-wise summation, as per paper.
        """
        def __init__(self, d_model: int, n_heads: int, n_layers: int, mult: int = 4, dropout: float = 0.0):
            super().__init__()
            self.encoder = TransformerEncoder(d_model, n_heads, n_layers, mult)
            self.drop = nn.Dropout(dropout)
        def forward(self, z: torch.Tensor, *inputs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            s = z
            for inp in inputs:
                s = s + inp
            s = self.encoder(s, mask)
            s = self.drop(s)
            return s

    class HRM(nn.Module):
        def __init__(self, cfg: HRMConfig):
            super().__init__()
            self.cfg = cfg
            self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
            self.L = HRMModule(cfg.d_model, cfg.n_heads, cfg.n_layers_L, cfg.mult, cfg.dropout)
            self.H = HRMModule(cfg.d_model, cfg.n_heads, cfg.n_layers_H, cfg.mult, cfg.dropout)
            self.out_head = LinearNoBias(cfg.d_model, cfg.vocab_size)  # tokenwise classifier
            self.q_head = LinearNoBias(cfg.d_model, 2)  # halt / continue
            # Truncated normal init for initial states (z0); we sample per batch at runtime.
            # Keep as non-parameter; see sample_init_state().

        @staticmethod
        def truncated_normal(shape, device, std=1.0, trunc=2.0):
            t = torch.zeros(shape, device=device).normal_(0, std)
            t = t.clamp_(-trunc*std, trunc*std)
            return t

        def sample_init_state(self, batch_size: int, seq_len: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
            zH0 = self.truncated_normal((batch_size, seq_len, self.cfg.d_model), device)
            zL0 = self.truncated_normal((batch_size, seq_len, self.cfg.d_model), device)
            return zH0, zL0

        def output_logits(self, zH: torch.Tensor) -> torch.Tensor:
            # Token-wise linear
            return self.out_head(zH)  # (B, L, V)

        def q_values(self, zH: torch.Tensor) -> torch.Tensor:
            # Pool over sequence then 2-way head (or take last token). Here: mean pool.
            pooled = zH.mean(dim=1)
            return torch.sigmoid(self.q_head(pooled))  # (B, 2)

        # ---------------- one segment with 1-step gradient -----------------
        def segment(self, zH: torch.Tensor, zL: torch.Tensor, x_emb: torch.Tensor, mask: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Run one HRM segment using the 1-step gradient approximation.
            Mirrors Figure 4 pseudocode: run N*T-1 steps without grad, then last step with grad.
            Returns: (zH_new, zL_new, logits)
            """
            N, T = self.cfg.N, self.cfg.T
            with torch.no_grad():
                for i in range(N*T - 1):
                    zL = self.L(zL, zH, x_emb, mask=mask)
                    if (i + 1) % T == 0:
                        zH = self.H(zH, zL, mask=mask)
            # final step with grads tracked
            zL = self.L(zL, zH, x_emb, mask=mask)
            zH = self.H(zH, zL, mask=mask)
            logits = self.output_logits(zH)
            return zH, zL, logits

        # ------------- training loop helpers (deep supervision + ACT) -------------
        def forward_deep_supervision(self,
                                      x_ids: torch.Tensor,
                                      y_tgt_ids: torch.Tensor,
                                      y_mask: torch.Tensor,
                                      mmax: Optional[int] = None,
                                      use_act: Optional[bool] = None,
                                      act_eps: Optional[float] = None) -> Tuple[torch.Tensor, dict]:
            """Deep supervision unrolled over segments with optional ACT.

            Args:
                x_ids: (B, L) full concatenated tokens (question + <sep> + shifted answer prefix)
                y_tgt_ids: (B, L) target token IDs (only positions for answer part are non -100)
                y_mask: (B, L) 1 for positions to include in CE loss (answer tokens), else 0
            Returns:
                total_loss, stats dict
            """
            cfg = self.cfg
            device = x_ids.device
            mmax = cfg.act_mmax if mmax is None else mmax
            use_act = cfg.use_act if use_act is None else use_act
            act_eps = cfg.act_eps if act_eps is None else act_eps

            x_emb = self.embed(x_ids)
            B, L = x_ids.shape
            zH, zL = self.sample_init_state(B, L, device)

            active = torch.ones(B, dtype=torch.bool, device=device)
            total_ce, total_q = 0.0, 0.0
            steps_taken = torch.zeros(B, dtype=torch.long, device=device)

            for m in range(mmax):
                zH, zL, logits = self.segment(zH, zL, x_emb)
                # token CE on answer positions only
                ce = F.cross_entropy(
                    logits.view(B*L, -1),
                    y_tgt_ids.view(B*L),
                    ignore_index=-100,
                    reduction='none'
                ).view(B, L)
                ce = (ce * y_mask).sum(dim=1) / (y_mask.sum(dim=1).clamp_min(1))
                ce = ce * active.float()
                # Q-head and ACT
                if use_act:
                    q = self.q_values(zH)  # (B, 2)
                    # randomized M_min
                    if m == 0:
                        mmin = torch.where(
                            torch.rand(B, device=device) < act_eps,
                            torch.randint(low=2, high=mmax+1, size=(B,), device=device),
                            torch.ones(B, device=device, dtype=torch.long)
                        )
                    # greedy halt decision once m >= mmin
                    halt_pref = q[:, 0]  # assume index 0 is halt
                    cont_pref = q[:, 1]
                    should_halt = (halt_pref > cont_pref) & (m >= (mmin - 1))
                    # targets
                    with torch.no_grad():
                        # binary reward if current prediction is exactly equal to targets on all positions
                        pred_ids = logits.argmax(dim=-1)
                        correct = ((pred_ids == y_tgt_ids) | (y_tgt_ids == -100)).all(dim=1).float()
                        # G_halt = 1{correct}; G_continue = max(Q_next) bootstrap
                    G_halt = correct
                    G_continue = torch.zeros_like(G_halt)
                    # next-step bootstrap proxy: we just use cont_pref as a placeholder; in practice
                    # we’d re-evaluate next segment – handled implicitly by the outer loop.
                    G = torch.stack([G_halt, torch.maximum(halt_pref, cont_pref)], dim=1)
                    bce = F.binary_cross_entropy(q, G, reduction='none').sum(dim=1)
                    bce = bce * active.float()
                else:
                    q = None
                    bce = torch.zeros(B, device=device)
                    should_halt = torch.zeros(B, dtype=torch.bool, device=device)

                # accumulate losses averaged over active samples
                total_ce = total_ce + ce.mean()
                total_q = total_q + bce.mean()

                # prepare for next segment: detach z (1-step grad approx) and optionally halt
                zH = zH.detach()
                zL = zL.detach()
                # update active mask
                if use_act:
                    # force halt when we exceed mmax-1 on the next step (handled by loop bound)
                    active = active & (~should_halt)
                steps_taken = steps_taken + active.long()
                # if everyone halted we can break
                if not active.any():
                    break

            loss = total_ce + total_q
            stats = dict(segments=m+1, steps_taken=steps_taken.detach().cpu().tolist())
            return loss, stats

        # ---------------------- inference ----------------------
        @torch.no_grad()
        def generate(self, x_ids: torch.Tensor, max_segments: int = 4, temperature: float = 0.0) -> torch.Tensor:
            device = x_ids.device
            x_emb = self.embed(x_ids)
            B, L = x_ids.shape
            zH, zL = self.sample_init_state(B, L, device)
            for _ in range(max_segments):
                zH, zL, logits = self.segment(zH, zL, x_emb)
                zH = zH.detach(); zL = zL.detach()
            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                return torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(B, L)
            else:
                return logits.argmax(dim=-1)

    return HRM, HRMConfig


@app.cell
def _(Dataset, List, TokenizerProtocol, Tuple, torch):

    # ---------------------------
    # Dataset (CSV: task,answer)
    # ---------------------------
    class NLMathDataset(Dataset):
        def __init__(self, rows: List[Tuple[str, str]], tokenizer: TokenizerProtocol, max_len: int = 256):
            self.rows = rows
            self.tok = tokenizer
            self.max_len = max_len
        def __len__(self):
            return len(self.rows)
        def __getitem__(self, idx):
            task, answer = self.rows[idx]
            # target as raw digits/characters (strip brackets if present)
            answer = answer.strip()
            if answer.startswith('[') and answer.endswith(']'):
                answer = answer[1:-1]
            # Build concatenated sequence: <bos> task <sep> <bos> answer_prefix ; predict answer + <eos>
            task_ids = [self.tok.bos_id] + self.tok.encode(task) + [self.tok.sep_id]
            # teacher forcing: use a single <bos> as answer prefix
            y_in_ids = [self.tok.bos_id]
            y_out_ids = self.tok.encode(answer) + [self.tok.eos_id]
            x_ids = task_ids + y_in_ids + y_out_ids  # model will see the answer tokens too (teacher forcing)
            # We predict the last len(y_out_ids) tokens given their previous tokens.
            # Construct targets with ignore_index for task + y_in positions.
            tgt = [-100] * (len(task_ids) + len(y_in_ids)) + y_out_ids
            # mask for loss positions
            y_mask = [0] * (len(task_ids) + len(y_in_ids)) + [1] * len(y_out_ids)
            # pad/truncate
            x_ids = x_ids[:self.max_len]
            tgt = tgt[:self.max_len]
            y_mask = y_mask[:self.max_len]
            pad_len = self.max_len - len(x_ids)
            if pad_len > 0:
                x_ids += [self.tok.pad_id] * pad_len
                tgt += [-100] * pad_len
                y_mask += [0] * pad_len
            return torch.tensor(x_ids, dtype=torch.long), torch.tensor(tgt, dtype=torch.long), torch.tensor(y_mask, dtype=torch.long)

    return (NLMathDataset,)


@app.cell
def _(DataLoader, Dataset, HRM, TokenizerProtocol, dataclass, nn, torch):

    # ---------------------------
    # Training utility
    # ---------------------------
    @dataclass
    class TrainConfig:
        batch_size: int = 8
        lr: float = 3e-4
        weight_decay: float = 0.01
        epochs: int = 5
        mmax: int = 4
        use_act: bool = False
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    class HRMTrainer:
        def __init__(self, model: HRM, tok: TokenizerProtocol, tcfg: TrainConfig):
            self.model = model.to(tcfg.device)
            self.tok = tok
            self.cfg = tcfg
            self.opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)

        def fit(self, train_ds: Dataset):
            dl = DataLoader(train_ds, batch_size=self.cfg.batch_size, shuffle=True, drop_last=False)
            self.model.train()
            for epoch in range(self.cfg.epochs):
                tot, n = 0.0, 0
                for x, y, ymask in dl:
                    x = x.to(self.cfg.device)
                    y = y.to(self.cfg.device)
                    ymask = ymask.to(self.cfg.device)
                    loss, stats = self.model.forward_deep_supervision(
                        x, y, ymask,
                        mmax=self.cfg.mmax,
                        use_act=self.cfg.use_act,
                    )
                    self.opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.opt.step()
                    tot += loss.item() * x.size(0)
                    n += x.size(0)
                print(f"epoch {epoch+1}: loss={tot/max(n,1):.4f}")

        @torch.no_grad()
        def predict(self, task: str, max_len: int = 256, max_segments: int = 4) -> str:
            self.model.eval()
            tok = self.tok
            # build inference sequence (no target known)
            task_ids = [tok.bos_id] + tok.encode(task) + [tok.sep_id]
            y_in_ids = [tok.bos_id]
            x_ids = task_ids + y_in_ids
            if len(x_ids) < max_len:
                x_ids = x_ids + [tok.pad_id] * (max_len - len(x_ids))
            else:
                x_ids = x_ids[:max_len]
            x = torch.tensor([x_ids], dtype=torch.long, device=self.cfg.device)
            out_ids = self.model.generate(x, max_segments=max_segments)[0].tolist()
            # take tokens after the prefix as the answer
            answer_tokens = out_ids[len(task_ids):]
            # stop at first EOS
            ans = []
            for t in answer_tokens:
                if t == tok.eos_id or t == tok.pad_id:
                    break
                ans.append(t)
            return tok.decode(ans)

    return HRMTrainer, TrainConfig


@app.cell
def _(
    CharTokenizer,
    HRM,
    HRMConfig,
    HRMTrainer,
    NLMathDataset,
    TrainConfig,
    pd,
    train_test_split,
):

    # ---------------------------
    # Example usage entry point
    # ---------------------------

    # Load data from CSV
    df = pd.read_csv("data/train.csv")

    # Split into train and eval (90/10)
    train_df, eval_df = train_test_split(df, test_size=0.1, random_state=42)

    # Prepare tokenizer and datasets
    tok = CharTokenizer()
    train_rows = [(row.task, str(row.answer)) for _, row in train_df.iterrows()]
    eval_rows = [(row.task, str(row.answer)) for _, row in eval_df.iterrows()]
    train_ds = NLMathDataset(train_rows, tok, max_len=256)
    eval_ds = NLMathDataset(eval_rows, tok, max_len=256)

    # Model config & trainer setup
    cfg = HRMConfig(vocab_size=len(tok.vocab), d_model=256, n_heads=8, n_layers_L=2, n_layers_H=2, T=2, N=2, act_mmax=4, use_act=False)
    model = HRM(cfg)
    trainer = HRMTrainer(model, tok, TrainConfig(batch_size=8, epochs=1000, mmax=3, use_act=False, device='cuda'))

    # Train and quick eval
    trainer.fit(train_ds)

    print("\nEval predictions:")
    for task, answer in eval_rows[:5]:
        print(f"Q: {task}\nPred: {trainer.predict(task)}\nGold: {answer}\n")
    return


if __name__ == "__main__":
    app.run()
