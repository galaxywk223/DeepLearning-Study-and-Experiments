from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .config import ExperimentConfig


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        *,
        embedding_dim: int,
        num_heads: int,
        block_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads.")

        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.qkv_proj = nn.Linear(embedding_dim, embedding_dim * 3)
        self.output_proj = nn.Linear(embedding_dim, embedding_dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)
        mask = torch.tril(torch.ones(block_size, block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, block_size, block_size))

    def forward(
        self,
        x: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, sequence_length, embedding_dim = x.shape
        qkv = self.qkv_proj(x)
        query, key, value = qkv.chunk(3, dim=-1)

        query = query.view(batch_size, sequence_length, self.num_heads, self.head_dim)
        key = key.view(batch_size, sequence_length, self.num_heads, self.head_dim)
        value = value.view(batch_size, sequence_length, self.num_heads, self.head_dim)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        attention_scores = (query @ key.transpose(-2, -1)) * (self.head_dim ** -0.5)
        mask = self.causal_mask[:, :, :sequence_length, :sequence_length]

        if attention_mask is not None:
            key_mask = attention_mask[:, None, None, :sequence_length]
            mask = mask & key_mask

        negative_infinity = torch.finfo(attention_scores.dtype).min
        attention_scores = attention_scores.masked_fill(~mask, negative_infinity)

        if attention_mask is not None:
            empty_rows = ~mask.any(dim=-1, keepdim=True)
            attention_scores = attention_scores.masked_fill(empty_rows, 0.0)

        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)

        if attention_mask is not None:
            query_mask = attention_mask[:, None, :, None].to(attention_weights.dtype)
            attention_weights = attention_weights * query_mask

        attended = attention_weights @ value
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size,
            sequence_length,
            embedding_dim,
        )
        attended = self.output_proj(attended)
        return self.residual_dropout(attended)


class FeedForward(nn.Module):
    def __init__(self, embedding_dim: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        hidden_dim = int(embedding_dim * mlp_ratio)
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        *,
        embedding_dim: int,
        num_heads: int,
        block_size: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.attention = CausalSelfAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            block_size=block_size,
            dropout=dropout,
        )
        self.ln2 = nn.LayerNorm(embedding_dim)
        self.feed_forward = FeedForward(
            embedding_dim=embedding_dim,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        x = x + self.attention(self.ln1(x), attention_mask=attention_mask)
        x = x + self.feed_forward(self.ln2(x))
        return x


class SubwordGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        *,
        pad_token_id: int,
        bos_token_id: int,
        block_size: int,
        embedding_dim: int,
        num_heads: int,
        num_layers: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.token_embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=pad_token_id,
        )
        self.position_embedding = nn.Embedding(block_size, embedding_dim)
        self.embedding_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    block_size=block_size,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size, bias=False)
        self.apply(self._init_weights)
        self.lm_head.weight = self.token_embedding.weight

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        token_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _, sequence_length = token_ids.shape
        if sequence_length > self.block_size:
            raise ValueError("Sequence length exceeds configured block_size.")

        positions = torch.arange(sequence_length, device=token_ids.device)
        token_embeddings = self.token_embedding(token_ids)
        position_embeddings = self.position_embedding(positions).unsqueeze(0)
        x = token_embeddings + position_embeddings

        if attention_mask is not None:
            x = x.masked_fill(~attention_mask.unsqueeze(-1), 0.0)

        x = self.embedding_dropout(x)
        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)
        x = self.final_norm(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(
        self,
        token_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        eos_token_id: int | None,
        allowed_token_ids: list[int] | None,
    ) -> torch.Tensor:
        finished = torch.zeros(token_ids.size(0), dtype=torch.bool, device=token_ids.device)

        for _ in range(max_new_tokens):
            context = token_ids[:, -self.block_size :]
            logits = self(context)[:, -1, :]
            logits = logits / max(temperature, 1e-5)

            if allowed_token_ids is not None:
                allowed_mask = torch.zeros(
                    logits.size(-1),
                    dtype=torch.bool,
                    device=logits.device,
                )
                allowed_mask[allowed_token_ids] = True
                logits[:, ~allowed_mask] = float("-inf")

            logits[:, self.pad_token_id] = float("-inf")
            logits[:, self.bos_token_id] = float("-inf")
            logits = apply_top_k(logits, top_k=top_k)
            logits = apply_top_p(logits, top_p=top_p)
            probabilities = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            token_ids = torch.cat((token_ids, next_token), dim=1)

            if eos_token_id is not None:
                finished |= next_token.squeeze(1) == eos_token_id
                if finished.all():
                    break

        return token_ids


def apply_top_k(logits: torch.Tensor, *, top_k: int) -> torch.Tensor:
    if top_k <= 0 or top_k >= logits.size(-1):
        return logits
    values, _ = torch.topk(logits, k=top_k, dim=-1)
    threshold = values[:, -1].unsqueeze(-1)
    return logits.masked_fill(logits < threshold, float("-inf"))


def apply_top_p(logits: torch.Tensor, *, top_p: float) -> torch.Tensor:
    if top_p <= 0.0 or top_p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probabilities = F.softmax(sorted_logits, dim=-1)
    cumulative_probabilities = torch.cumsum(sorted_probabilities, dim=-1)

    sorted_mask = cumulative_probabilities > top_p
    sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
    sorted_mask[:, 0] = False

    mask = torch.zeros_like(sorted_mask, dtype=torch.bool)
    mask.scatter_(1, sorted_indices, sorted_mask)
    return logits.masked_fill(mask, float("-inf"))


def build_model(
    *,
    vocab_size: int,
    pad_token_id: int,
    bos_token_id: int,
    config: ExperimentConfig,
) -> nn.Module:
    if config.variant != "gpt":
        raise ValueError(f"Unsupported variant: {config.variant}")

    return SubwordGPT(
        vocab_size=vocab_size,
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        block_size=config.block_size,
        embedding_dim=config.embedding_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        mlp_ratio=config.mlp_ratio,
        dropout=config.dropout,
    )
