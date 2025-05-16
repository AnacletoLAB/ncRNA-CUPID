import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional
from transformers import PreTrainedModel, PretrainedConfig

# Fallback import for ModelOutput across HF versions
try:
    from transformers import ModelOutput
except ImportError:
    from transformers.modeling_outputs import ModelOutput

class RNACrossAttentionConfig(PretrainedConfig):
    model_type = "rna-cross-attention"

    def __init__(
        self,
        d_model: int = 1024,
        n_heads: int = 8,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = dropout

@dataclass
class RNACrossAttentionOutput(ModelOutput):
    """
    Output of RNACrossAttentionHF.
    *loss* must come first so Trainer picks it up.
    """
    loss: Optional[torch.Tensor] = None
    logits: torch.Tensor = None

class RNACrossAttentionHF(PreTrainedModel):
    config_class = RNACrossAttentionConfig

    def __init__(self, config: RNACrossAttentionConfig):
        super().__init__(config)
        d_model = config.d_model
        n_heads = config.n_heads
        dropout = config.dropout

        # Self-attention for BoM tokens (shared for A and B)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True
        )
        # Cross-attention for BoM tokens
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True
        )
        # Downstream MLP classifier on concatenated pooled vectors
        self.classifier = nn.Sequential(
            nn.Linear(2 * d_model, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        self.loss_fn = nn.BCEWithLogitsLoss()

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        A: torch.Tensor,                # [batch, lenA, d_model]
        B: torch.Tensor,                # [batch, lenB, d_model]
        maskA: Optional[torch.Tensor] = None,  # [batch, lenA] True = pad
        maskB: Optional[torch.Tensor] = None,  # [batch, lenB] True = pad
        labels: Optional[torch.Tensor] = None  # [batch]
    ) -> RNACrossAttentionOutput:
        # 1. Self-attention over each sequence
        A_self, _ = self.self_attn(A, A, A, key_padding_mask=maskA)
        B_self, _ = self.self_attn(B, B, B, key_padding_mask=maskB)

        # 2. Cross-attention pooling: A conditioned on B (values from B)
        attn_AB, _ = self.cross_attn(
            query=A_self, key=B_self, value=B_self,
            key_padding_mask=maskB
        )  # [batch, lenA, d_model]
        z_AB = self._masked_mean(attn_AB, maskA)

        # 3. Cross-attention pooling: B conditioned on A (values from A)
        attn_BA, _ = self.cross_attn(
            query=B_self, key=A_self, value=A_self,
            key_padding_mask=maskA
        )  # [batch, lenB, d_model]
        z_BA = self._masked_mean(attn_BA, maskB)

        # 4. Concat and classify
        pooled = torch.cat([z_AB, z_BA], dim=-1)  # [batch, 2*d_model]
        logits = self.classifier(pooled).squeeze(-1)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels.float())

        return RNACrossAttentionOutput(loss=loss, logits=logits)

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Mean-pools over the sequence dimension of x, ignoring padded positions.
        x: [batch, len, d], mask: [batch, len] (True = pad)
        """
        if mask is not None:
            valid = ~mask                        # [batch, len]
            valid = valid.unsqueeze(-1).float()  # [batch, len, 1]
            x = x * valid                       # zero-out pads
            summed = x.sum(dim=1)              # [batch, d]
            count = valid.sum(dim=1).clamp(min=1e-8)  # [batch, 1]
            return summed / count
        else:
            return x.mean(dim=1)

