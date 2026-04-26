"""Information Bottleneck ARC Model v2.

Improvements over v1:
  - Multi-demo aggregation: average rule vectors from N demo pairs
  - Optional cross-attention decoder (rule attends to test)
  - Configurable encoder: can use deterministic projection instead of VAE
"""

from typing import Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed


class RuleEncoderV2(nn.Module):
    """Encodes (input, output) demo pairs into a rule vector.
    
    Supports multi-demo: encode each pair independently, then aggregate.
    """

    def __init__(
        self,
        image_size: int = 64,
        num_colors: int = 12,
        embed_dim: int = 256,
        depth: int = 4,
        num_heads: int = 8,
        rule_dim: int = 32,
        patch_size: int = 2,
        use_vae: bool = True,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_colors = num_colors
        self.embed_dim = embed_dim
        self.rule_dim = rule_dim
        self.use_vae = use_vae
        seq_len = (image_size // patch_size) ** 2

        self.color_embed = nn.Embedding(num_colors, embed_dim)
        self.input_patch = PatchEmbed(image_size, patch_size, embed_dim, embed_dim, bias=True)
        self.output_patch = PatchEmbed(image_size, patch_size, embed_dim, embed_dim, bias=True)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        self.role_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=0.1, activation='gelu',
                batch_first=True, norm_first=True,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        if use_vae:
            self.rule_mu = nn.Linear(embed_dim, rule_dim)
            self.rule_logvar = nn.Linear(embed_dim, rule_dim)
        else:
            self.rule_proj = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, rule_dim),
            )

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.role_embed, std=0.02)

    def _embed_grid(self, grid, patch_fn):
        tokens = self.color_embed(grid.long())
        tokens = patch_fn(tokens.permute(0, 3, 1, 2))
        tokens = tokens + self.pos_embed[:, :tokens.size(1), :]
        return tokens

    def encode_single(self, demo_input, demo_output):
        in_tokens = self._embed_grid(demo_input, self.input_patch)
        out_tokens = self._embed_grid(demo_output, self.output_patch)
        out_tokens = out_tokens + self.role_embed
        tokens = torch.cat([in_tokens, out_tokens], dim=1)
        for layer in self.layers:
            tokens = layer(tokens)
        tokens = self.norm(tokens)
        return tokens.mean(dim=1)

    def forward(self, demo_inputs, demo_outputs):
        """
        Args:
            demo_inputs:  (B, N, H, W) or (B, H, W) — N demo pairs per task
            demo_outputs: (B, N, H, W) or (B, H, W)
        Returns:
            rule_vector: (B, rule_dim)
            kl_loss: scalar
        """
        if demo_inputs.dim() == 3:
            demo_inputs = demo_inputs.unsqueeze(1)
            demo_outputs = demo_outputs.unsqueeze(1)
        
        B, N, H, W = demo_inputs.shape
        
        # Encode each demo pair independently
        all_pooled = []
        for i in range(N):
            pooled = self.encode_single(demo_inputs[:, i], demo_outputs[:, i])
            all_pooled.append(pooled)
        
        # Aggregate: mean pooling over demo pairs
        pooled = torch.stack(all_pooled, dim=1).mean(dim=1)  # (B, embed_dim)

        if self.use_vae:
            mu = self.rule_mu(pooled)
            logvar = self.rule_logvar(pooled)
            if self.training:
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                rule = mu + eps * std
            else:
                rule = mu
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
        else:
            rule = self.rule_proj(pooled)
            kl = torch.tensor(0.0, device=rule.device)

        return rule, kl


class CrossAttentionDecoder(nn.Module):
    """Decoder using cross-attention: test tokens attend to rule.
    
    More expressive than simple rule injection via prepended token.
    """

    def __init__(
        self,
        image_size: int = 64,
        num_colors: int = 12,
        embed_dim: int = 256,
        rule_dim: int = 32,
        num_heads: int = 8,
        num_iterations: int = 4,
        depth: int = 2,
        patch_size: int = 2,
    ):
        super().__init__()
        self.image_size = image_size
        self.num_colors = num_colors
        self.embed_dim = embed_dim
        self.num_iterations = num_iterations
        self.patch_size = patch_size
        self.seq_len = (image_size // patch_size) ** 2

        self.color_embed = nn.Embedding(num_colors, embed_dim)
        self.patch_embed = PatchEmbed(image_size, patch_size, embed_dim, embed_dim, bias=True)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len, embed_dim))

        # Rule expansion: single vector -> K rule tokens for richer cross-attention
        self.rule_expand = nn.Sequential(
            nn.Linear(rule_dim, embed_dim * 4),
            nn.GELU(),
            nn.Unflatten(1, (4, embed_dim)),
        )

        # Self-attention + cross-attention per iteration (shared weights)
        self.self_attn = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=0.1, activation='gelu',
                batch_first=True, norm_first=True,
            )
            for _ in range(depth)
        ])
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads,
            batch_first=True, dropout=0.1,
        )
        self.cross_norm = nn.LayerNorm(embed_dim)
        self.cross_ff = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        
        self.update_gate = nn.Linear(embed_dim, embed_dim)
        self.head = nn.Linear(embed_dim, num_colors * patch_size ** 2)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _embed_grid(self, grid):
        tokens = self.color_embed(grid.long())
        tokens = self.patch_embed(tokens.permute(0, 3, 1, 2))
        tokens = tokens + self.pos_embed[:, :tokens.size(1), :]
        return tokens

    def forward(self, test_input, rule_vector):
        B = test_input.size(0)
        state = self._embed_grid(test_input)
        rule_tokens = self.rule_expand(rule_vector)  # (B, 4, embed_dim)

        for iteration in range(self.num_iterations):
            # Self-attention
            for layer in self.self_attn:
                state = layer(state)
            
            # Cross-attention to rule tokens
            residual = state
            state_normed = self.cross_norm(state)
            attended, _ = self.cross_attn(state_normed, rule_tokens, rule_tokens)
            state = residual + attended
            state = state + self.cross_ff(state)
            
            # Gated update
            if iteration < self.num_iterations - 1:
                gate = torch.sigmoid(self.update_gate(state))
                state = residual * (1 - gate) + state * gate

        logits = self.head(state)
        ps = self.patch_size
        logits = logits.reshape(B, self.image_size // ps, self.image_size // ps, ps, ps, self.num_colors)
        logits = logits.permute(0, 1, 3, 2, 4, 5)
        logits = logits.reshape(B, self.image_size, self.image_size, self.num_colors)
        logits = logits.permute(0, 3, 1, 2)
        return logits


class ARCBottleneckV2(nn.Module):
    """V2: multi-demo, cross-attention decoder, optional VAE."""

    def __init__(
        self,
        image_size: int = 64,
        num_colors: int = 12,
        embed_dim: int = 256,
        encoder_depth: int = 4,
        decoder_depth: int = 2,
        num_heads: int = 8,
        rule_dim: int = 32,
        num_iterations: int = 4,
        patch_size: int = 2,
        kl_weight: float = 0.001,
        use_vae: bool = True,
    ):
        super().__init__()
        self.kl_weight = kl_weight
        self.rule_dim = rule_dim

        self.encoder = RuleEncoderV2(
            image_size=image_size, num_colors=num_colors,
            embed_dim=embed_dim, depth=encoder_depth,
            num_heads=num_heads, rule_dim=rule_dim,
            patch_size=patch_size, use_vae=use_vae,
        )
        self.decoder = CrossAttentionDecoder(
            image_size=image_size, num_colors=num_colors,
            embed_dim=embed_dim, rule_dim=rule_dim,
            num_heads=num_heads, num_iterations=num_iterations,
            depth=decoder_depth, patch_size=patch_size,
        )

    def forward(self, demo_input, demo_output, test_input, attention_mask=None):
        rule, kl_loss = self.encoder(demo_input, demo_output)
        logits = self.decoder(test_input, rule)
        return logits, kl_loss

    def encode_rule(self, demo_input, demo_output):
        with torch.no_grad():
            rule, _ = self.encoder(demo_input, demo_output)
        return rule

    @torch.no_grad()
    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        encoder = sum(p.numel() for p in self.encoder.parameters())
        decoder = sum(p.numel() for p in self.decoder.parameters())
        return {"total": total, "encoder": encoder, "decoder": decoder}
