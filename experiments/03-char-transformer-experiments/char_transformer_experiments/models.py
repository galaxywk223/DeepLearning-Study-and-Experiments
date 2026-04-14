from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .config import ExperimentConfig


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, block_size: int) -> None:
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.token_embedding(token_ids)

    @torch.no_grad()
    def generate(
        self,
        token_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        temperature: float,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            context = token_ids[:, -self.block_size :]
            logits = self(context)[:, -1, :]
            logits = logits / max(temperature, 1e-5)
            probabilities = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            token_ids = torch.cat((token_ids, next_token), dim=1)
        return token_ids


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        attention_scores = attention_scores.masked_fill(~mask, float("-inf"))
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))
        return x


class CharTransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        *,
        block_size: int,
        embedding_dim: int,
        num_heads: int,
        num_layers: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(block_size, embedding_dim)
        self.embedding_dropout = nn.Dropout(dropout)
        self.blocks = nn.Sequential(
            *[
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
        self.lm_head = nn.Linear(embedding_dim, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        _, sequence_length = token_ids.shape
        if sequence_length > self.block_size:
            raise ValueError("Sequence length exceeds configured block_size.")

        positions = torch.arange(sequence_length, device=token_ids.device)
        token_embeddings = self.token_embedding(token_ids)
        position_embeddings = self.position_embedding(positions).unsqueeze(0)
        x = token_embeddings + position_embeddings
        x = self.embedding_dropout(x)
        x = self.blocks(x)
        x = self.final_norm(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(
        self,
        token_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        temperature: float,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            context = token_ids[:, -self.block_size :]
            logits = self(context)[:, -1, :]
            logits = logits / max(temperature, 1e-5)
            probabilities = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            token_ids = torch.cat((token_ids, next_token), dim=1)
        return token_ids


def build_model(
    variant: str,
    *,
    vocab_size: int,
    config: ExperimentConfig,
) -> nn.Module:
    normalized = variant.lower()
    if normalized == "bigram":
        return BigramLanguageModel(vocab_size=vocab_size, block_size=config.block_size)
    if normalized == "transformer":
        return CharTransformerLM(
            vocab_size=vocab_size,
            block_size=config.block_size,
            embedding_dim=config.embedding_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout,
        )
    raise ValueError(f"Unsupported variant: {variant}")
