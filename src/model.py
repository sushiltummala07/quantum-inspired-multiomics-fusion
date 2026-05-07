"""
model.py — Full Quantum Multi-Omics Fusion Model.

Components:
  • Three independent QuantumEncoders (one per modality)
  • Transformer-style cross-modal attention fusion
  • Modality dropout (randomly mask one modality during training)
  • Batch-normed MLP classifier
  • Optional survival / regression head (multi-task)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from quantum import QuantumEncoder
from config import QuantumConfig, FusionConfig


# ──────────────────────────────────────────────
# Positional / modality embeddings
# ──────────────────────────────────────────────
class ModalityEmbedding(nn.Module):
    """Learned token to distinguish genomics / transcriptomics / proteomics."""
    def __init__(self, embed_dim: int, n_modalities: int = 3):
        super().__init__()
        self.emb = nn.Embedding(n_modalities, embed_dim)

    def forward(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        x = x.float()  #ensure consistency of dtype
        tok = self.emb(torch.tensor(idx, device=x.device, dtype=torch.long)).unsqueeze(0)
        tok = tok.to(x.dtype)
        return x + tok


# ──────────────────────────────────────────────
# Cross-modal Transformer Fusion
# ──────────────────────────────────────────────
class CrossModalFusion(nn.Module):
    """
    Stack of Transformer encoder layers operating over the three modality tokens.
    Returns the fused representation + per-sample attention weights for analysis.
    """
    def __init__(self, embed_dim: int, n_heads: int = 2,
                 n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.mod_emb = ModalityEmbedding(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True      # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 3, 3),
            nn.Softmax(dim=-1)
        )

    def forward(self, g, t, p):
        # 🔥 Ensure consistent dtype
        g = g.float()
        t = t.float()
        p = p.float()
        # Add modality tokens
        g = self.mod_emb(g, 0)
        t = self.mod_emb(t, 1)
        p = self.mod_emb(p, 2)

        seq = torch.stack([g, t, p], dim=1).float()            # (B, 3, D)
        out = self.transformer(seq)                      # (B, 3, D)

        # Gated pooling
        flat  = out.reshape(out.size(0), -1)             # (B, 3D)
        gates = self.gate(flat)                          # (B, 3)
        fused = (out * gates.unsqueeze(-1)).sum(dim=1)   # (B, D)

        return fused, gates


# ──────────────────────────────────────────────
# Full Model
# ──────────────────────────────────────────────
class QuantumMultiOmicsFusion(nn.Module):
    """
    Full quantum-classical hybrid model for multi-omics disease prediction.

    Args:
        qcfg       : QuantumConfig
        fcfg       : FusionConfig
        n_classes  : number of output classes
        survival   : if True, add auxiliary Cox proportional hazard output
    """
    def __init__(self, qcfg: QuantumConfig, fcfg: FusionConfig,
                 n_classes: int = 2, survival: bool = False):
        super().__init__()
        D = qcfg.n_qubits

        # Quantum encoders (one per modality)
        self.enc_g = QuantumEncoder(qcfg)
        self.enc_t = QuantumEncoder(qcfg)
        self.enc_p = QuantumEncoder(qcfg)

        # Fusion
        self.fusion = CrossModalFusion(
            embed_dim=D, n_heads=fcfg.n_heads,
            n_layers=2, dropout=fcfg.dropout
        )

        # Classification head
        self.bn = nn.BatchNorm1d(D)
        self.classifier = nn.Sequential(
            nn.Linear(D, D * 4),
            nn.GELU(),
            nn.Dropout(fcfg.dropout),
            nn.Linear(D * 4, D * 2),
            nn.GELU(),
            nn.Dropout(fcfg.dropout),
            nn.Linear(D * 2, n_classes)
        )

        # Optional survival head (outputs log-hazard ratio)
        self.survival = survival
        if survival:
            self.survival_head = nn.Sequential(
                nn.Linear(D, D),
                nn.Tanh(),
                nn.Linear(D, 1)
            )

        self.use_modality_dropout = fcfg.use_modality_dropout
        self.D = D

    def _modality_dropout(self, g, t, p, p_drop=0.2):
        """Randomly drop each modality independently per sample."""
        if not self.training:
            return g, t, p

        B = g.shape[0]
        device = g.device

        # Create independent masks for each modality
        mask = (torch.rand(B, 3, device=device) > p_drop).float()

        g = g * mask[:, 0].unsqueeze(-1)
        t = t * mask[:, 1].unsqueeze(-1)
        p = p * mask[:, 2].unsqueeze(-1)

        return g, t, p

    def forward(self, g, t, p, return_aux=False):
        # Quantum encoding
        ge = self.enc_g(g)
        te = self.enc_t(t)
        pe = self.enc_p(p)

        if self.use_modality_dropout:
            ge, te, pe = self._modality_dropout(ge, te, pe)

        # Fusion
        fused, gates = self.fusion(ge, te, pe)
        fused_bn = self.bn(fused)
        logits = self.classifier(fused_bn)

        if return_aux:
            aux = {"gates": gates, "embedding": fused}
            if self.survival:
                aux["log_hazard"] = self.survival_head(fused_bn).squeeze(-1)
            return logits, aux

        return logits

    def encode(self, g, t, p) -> torch.Tensor:
        """Return fused embedding (no classification head)."""
        with torch.no_grad():
            ge = self.enc_g(g)
            te = self.enc_t(t)
            pe = self.enc_p(p)
            fused, _ = self.fusion(ge, te, pe)
        return fused
