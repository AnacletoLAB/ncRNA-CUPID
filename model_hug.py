## model_hug.py

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
    ModelOutput for RNACrossAttentionHF. 
    *loss* must come first so Trainer picks it up.
    """
    loss: Optional[torch.Tensor] = None
    logits: torch.Tensor = None


class RNACrossAttentionHF(PreTrainedModel):
    config_class = RNACrossAttentionConfig

    def __init__(self, config: RNACrossAttentionConfig):
        super().__init__(config)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.n_heads,
            dropout=config.dropout,
            batch_first=True
        )
        self.classifier = nn.Linear(config.d_model, 1)
        self.post_init()

    def forward(
        self,
        A: torch.Tensor,               # [batch, lenA, d_model]
        B: torch.Tensor,               # [batch, lenB, d_model]
        maskA: Optional[torch.Tensor] = None,  # [batch, lenA]
        maskB: Optional[torch.Tensor] = None,  # [batch, lenB]
        labels: Optional[torch.Tensor] = None  # [batch]
    ) -> RNACrossAttentionOutput:
        # Cross‐attention
        attn_output, _ = self.cross_attn(
            query=A,
            key=B,
            value=B,
            key_padding_mask=maskB
        )  # [batch, lenA, d_model]

        # Masked mean‐pool over A
        if maskA is not None:
            valid = (~maskA).unsqueeze(-1)              # [batch, lenA, 1]
            summed = (attn_output * valid).sum(dim=1)   # [batch, d_model]
            counts = valid.sum(dim=1).clamp(min=1)      # [batch, 1]
            pooled = summed / counts                    # [batch, d_model]
        else:
            pooled = attn_output.mean(dim=1)            # [batch, d_model]

        logits = self.classifier(pooled).squeeze(-1)    # [batch]
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels.float())

        return RNACrossAttentionOutput(loss=loss, logits=logits)
