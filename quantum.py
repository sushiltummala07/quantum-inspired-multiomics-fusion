"""
quantum.py — Expressive quantum encoders.

Encoding strategies:
  • "angle"     : RX angle encoding (simple, good baseline)
  • "amplitude" : Amplitude encoding via RY/RZ (more expressive)
  • "iqp"       : Instantaneous Quantum Polynomial (IQP) kernel encoding

Variational ansatz: Hardware-Efficient Ansatz (HEA) with configurable
entanglement topology (circular | full | linear).
"""

import torch
import torch.nn as nn
import pennylane as qml
import numpy as np
from config import QuantumConfig


# ──────────────────────────────────────────────
# Circuit builders
# ──────────────────────────────────────────────
def _entangle(n_qubits: int, topology: str):
    """Apply entanglement layer."""
    if topology == "full":
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                qml.CZ(wires=[i, j])
    elif topology == "circular":
        for i in range(n_qubits):
            qml.CNOT(wires=[i, (i + 1) % n_qubits])
    else:  # linear
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])


def build_circuit(qcfg: QuantumConfig):
    """
    Returns a QNode that encodes `inputs` (shape: n_qubits) and applies
    `n_layers` variational HEA layers. Weights shape: (n_layers, n_qubits, 3).
    """
    n  = qcfg.n_qubits
    nl = qcfg.n_layers
    enc = qcfg.encoding
    ent = qcfg.entanglement
    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev, interface="torch", diff_method=qcfg.diff_method)
    def circuit(inputs, weights):
        # ── Encoding ──────────────────────────────
        if enc == "angle":
            for i in range(n):
                qml.RX(inputs[i], wires=i)

        elif enc == "amplitude":
            # Encode via RY(arctan(x)) + RZ(arctan(x²))
            for i in range(n):
                qml.RY(torch.arctan(inputs[i]), wires=i)
                qml.RZ(torch.arctan(inputs[i] ** 2), wires=i)

        elif enc == "iqp":
            # IQP: Hadamard → RZ(x) → ZZ interactions
            for i in range(n):
                qml.Hadamard(wires=i)
                qml.RZ(inputs[i], wires=i)
            for i in range(n - 1):
                qml.IsingZZ(inputs[i] * inputs[i + 1], wires=[i, i + 1])

        # ── Variational HEA layers ────────────────
        for layer in range(nl):
            for i in range(n):
                qml.Rot(weights[layer, i, 0],
                        weights[layer, i, 1],
                        weights[layer, i, 2], wires=i)
            _entangle(n, ent)

        # ── Measurement ──────────────────────────
        return [qml.expval(qml.PauliZ(i)) for i in range(n)]

    return circuit


# ──────────────────────────────────────────────
# PyTorch module
# ──────────────────────────────────────────────
class QuantumEncoder(nn.Module):
    """
    Wraps a PennyLane variational circuit.
    Input  : (batch, n_qubits)
    Output : (batch, n_qubits)  — expectation values ∈ [-1, 1]
    """
    def __init__(self, qcfg: QuantumConfig):
        super().__init__()
        self.n_qubits = qcfg.n_qubits
        self.circuit  = build_circuit(qcfg)
        # Glorot-style init scaled for rotation gates
        self.weights  = nn.Parameter(
            torch.empty(qcfg.n_layers, qcfg.n_qubits, 3).uniform_(
                -np.pi / 4, np.pi / 4
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process batch sample-by-sample (PennyLane limitation)."""
        out = torch.stack([
            torch.stack(self.circuit(x[i], self.weights))
            for i in range(x.shape[0])
        ])                              # (B, n_qubits)
        return out


# ──────────────────────────────────────────────
# Contrastive pre-training loss
# ──────────────────────────────────────────────
class NTXentLoss(nn.Module):
    """
    Normalised temperature-scaled cross-entropy loss (SimCLR).
    Used to pre-train each quantum encoder in a self-supervised fashion
    by treating augmented views of the same sample as positives.
    """
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.T = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        B    = z1.shape[0]
        z    = torch.cat([z1, z2], dim=0)                     # (2B, D)
        z    = nn.functional.normalize(z, dim=-1)
        sim  = (z @ z.T) / self.T                              # (2B, 2B)
        # Remove self-similarity
        mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, -1e9)
        # Positive pairs: (i, i+B) and (i+B, i)
        pos  = torch.cat([torch.diag(sim, B), torch.diag(sim, -B)])  # (2B,)
        loss = -pos + torch.logsumexp(sim, dim=-1)
        return loss.mean()


def pretrain_encoder(encoder: QuantumEncoder, X: torch.Tensor,
                     epochs: int = 20, lr: float = 1e-2,
                     noise_std: float = 0.05) -> QuantumEncoder:
    """
    Self-supervised contrastive pre-training via two augmented views.
    Augmentation: additive Gaussian noise (quantum-measurement-like noise).
    """
    encoder.train()
    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    criterion = NTXentLoss(temperature=0.1)

    for epoch in range(epochs):
        # Two noisy views
        x1 = X + noise_std * torch.randn_like(X)
        x2 = X + noise_std * torch.randn_like(X)

        optimizer.zero_grad()
        z1 = encoder(x1)
        z2 = encoder(x2)
        loss = criterion(z1, z2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            print(f"  [Pretrain] epoch {epoch+1:3d}  loss={loss.item():.4f}")

    return encoder
