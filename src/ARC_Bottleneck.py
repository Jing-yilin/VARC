"""
Information Bottleneck ARC Model.

Core idea from first principles:
  - The universe's rules are simple (low Kolmogorov complexity)
  - Understanding = compression (find the shortest program)
  - Rules are local and iteratively applied

Architecture:
  1. Rule Encoder: sees demo (input, output) pairs → compresses to rule_dim vector
  2. Rule Decoder: takes test_input + rule_vector → iteratively produces output
  3. Information bottleneck forces the rule vector to be low-dimensional (~32d)
     so the model MUST discover the abstract rule, not memorize pixels
"""

from typing import Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed
from utils.pos_embed import VisionRotaryEmbeddingFast


class RuleEncoder(nn.Module):
    """Encodes (input, output) demo pairs into a low-dimensional rule vector.

    The encoder sees both input and output simultaneously — this is what
    VARC's original architecture CANNOT do. It enables the model to compute
    the diff (what changed?) which is the essence of rule discovery.
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
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_colors = num_colors
        self.embed_dim = embed_dim
        self.rule_dim = rule_dim
        seq_len = (image_size // patch_size) ** 2

        self.color_embed = nn.Embedding(num_colors, embed_dim)
        self.input_patch = PatchEmbed(image_size, patch_size, embed_dim, embed_dim, bias=True)
        self.output_patch = PatchEmbed(image_size, patch_size, embed_dim, embed_dim, bias=True)

        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, embed_dim))

        self.role_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 2,
                dropout=0.0,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self.rule_mu = nn.Linear(embed_dim, rule_dim)
        self.rule_logvar = nn.Linear(embed_dim, rule_dim)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.role_embed, std=0.02)

    def _embed_grid(self, grid, patch_fn):
        tokens = self.color_embed(grid.long())
        tokens = patch_fn(tokens.permute(0, 3, 1, 2))
        tokens = tokens + self.pos_embed[:, :tokens.size(1), :]
        return tokens

    def forward(self, demo_input, demo_output):
        """
        Args:
            demo_input:  (B, H, W) integer grid
            demo_output: (B, H, W) integer grid
        Returns:
            rule_vector: (B, rule_dim)
            kl_loss: scalar, KL divergence for the bottleneck
        """
        in_tokens = self._embed_grid(demo_input, self.input_patch)
        out_tokens = self._embed_grid(demo_output, self.output_patch)
        out_tokens = out_tokens + self.role_embed

        tokens = torch.cat([in_tokens, out_tokens], dim=1)

        for layer in self.layers:
            tokens = layer(tokens)
        tokens = self.norm(tokens)

        pooled = tokens.mean(dim=1)

        mu = self.rule_mu(pooled)
        logvar = self.rule_logvar(pooled)

        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            rule = mu + eps * std
        else:
            rule = mu

        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()

        return rule, kl


class IterativeRuleDecoder(nn.Module):
    """Applies a rule vector to a test input through iterative local updates.

    Like a cellular automaton: the same local rule is applied repeatedly
    until convergence. This mirrors how the universe works — simple local
    rules, iterated many times, produce complex global patterns.
    """

    def __init__(
        self,
        image_size: int = 64,
        num_colors: int = 12,
        embed_dim: int = 256,
        rule_dim: int = 32,
        num_heads: int = 8,
        num_iterations: int = 4,
        patch_size: int = 2,
    ):
        super().__init__()
        self.image_size = image_size
        self.num_colors = num_colors
        self.embed_dim = embed_dim
        self.rule_dim = rule_dim
        self.num_iterations = num_iterations
        self.patch_size = patch_size
        self.seq_len = (image_size // patch_size) ** 2

        self.color_embed = nn.Embedding(num_colors, embed_dim)
        self.patch_embed = PatchEmbed(image_size, patch_size, embed_dim, embed_dim, bias=True)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len, embed_dim))

        self.rule_proj = nn.Linear(rule_dim, embed_dim)

        self.update_block = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 2,
                dropout=0.0,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            )
            for _ in range(2)
        ])
        self.update_norm = nn.LayerNorm(embed_dim)
        self.update_gate = nn.Linear(embed_dim, embed_dim)

        self.head = nn.Linear(embed_dim, num_colors * patch_size ** 2)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _embed_grid(self, grid):
        tokens = self.color_embed(grid.long())
        tokens = self.patch_embed(tokens.permute(0, 3, 1, 2))
        tokens = tokens + self.pos_embed[:, :tokens.size(1), :]
        return tokens

    def forward(self, test_input, rule_vector):
        """
        Args:
            test_input:   (B, H, W) integer grid
            rule_vector:  (B, rule_dim)
        Returns:
            logits: (B, num_colors, H, W)
        """
        B = test_input.size(0)
        state = self._embed_grid(test_input)
        rule_token = self.rule_proj(rule_vector).unsqueeze(1)

        for iteration in range(self.num_iterations):
            tokens = torch.cat([rule_token, state], dim=1)

            for layer in self.update_block:
                tokens = layer(tokens)
            tokens = self.update_norm(tokens)

            updated = tokens[:, 1:, :]

            gate = torch.sigmoid(self.update_gate(updated))
            state = state * (1 - gate) + updated * gate

        logits = self.head(state)
        ps = self.patch_size
        logits = logits.reshape(B, self.image_size // ps, self.image_size // ps, ps, ps, self.num_colors)
        logits = logits.permute(0, 1, 3, 2, 4, 5)
        logits = logits.reshape(B, self.image_size, self.image_size, self.num_colors)
        logits = logits.permute(0, 3, 1, 2)
        return logits


class ARCBottleneck(nn.Module):
    """Full model: Rule Encoder + Iterative Decoder with information bottleneck.

    The bottleneck (rule_dim << embed_dim) forces compression:
    - 512-dim task token (original VARC) = 16,384 bits = can memorize pixels
    - 32-dim rule vector (ours) = 1,024 bits = must discover rules
    """

    def __init__(
        self,
        image_size: int = 64,
        num_colors: int = 12,
        embed_dim: int = 256,
        encoder_depth: int = 4,
        num_heads: int = 8,
        rule_dim: int = 32,
        num_iterations: int = 4,
        patch_size: int = 2,
        kl_weight: float = 0.001,
    ):
        super().__init__()
        self.kl_weight = kl_weight
        self.rule_dim = rule_dim

        self.encoder = RuleEncoder(
            image_size=image_size,
            num_colors=num_colors,
            embed_dim=embed_dim,
            depth=encoder_depth,
            num_heads=num_heads,
            rule_dim=rule_dim,
            patch_size=patch_size,
        )
        self.decoder = IterativeRuleDecoder(
            image_size=image_size,
            num_colors=num_colors,
            embed_dim=embed_dim,
            rule_dim=rule_dim,
            num_heads=num_heads,
            num_iterations=num_iterations,
            patch_size=patch_size,
        )

    def forward(
        self,
        demo_input: torch.Tensor,
        demo_output: torch.Tensor,
        test_input: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            demo_input:  (B, H, W) demo input grid
            demo_output: (B, H, W) demo output grid
            test_input:  (B, H, W) test input grid
        Returns:
            logits: (B, num_colors, H, W)
            kl_loss: scalar
        """
        rule, kl_loss = self.encoder(demo_input, demo_output)
        logits = self.decoder(test_input, rule)
        return logits, kl_loss

    def encode_rule(self, demo_input, demo_output):
        """Extract rule vector only (for analysis/visualization)."""
        with torch.no_grad():
            rule, _ = self.encoder(demo_input, demo_output)
        return rule

    @torch.no_grad()
    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        encoder = sum(p.numel() for p in self.encoder.parameters())
        decoder = sum(p.numel() for p in self.decoder.parameters())
        return {"total": total, "encoder": encoder, "decoder": decoder}
