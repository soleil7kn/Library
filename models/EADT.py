import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted, PositionalEmbedding


class SpectralEntropyEstimator(nn.Module):
    """SEED-style learnable spectral-entropy estimator.

    Args:
        seq_len: Input sequence length.
        normalized: If True, normalize entropy by log2(number of frequency bins).
        eps: Numerical-stability constant.

    Input:
        x: [B, C, T]

    Output:
        normalized spectral entropy: [B, C], approximately in [0, 1].

    This follows the learnable frequency-filter calculation used in the
    supplied spectral-entropy implementation: the input spectrum is multiplied
    by the FFT of a learnable time-domain filter before the power spectrum and
    entropy are calculated.
    """

    def __init__(self, seq_len, normalized=True, eps=1e-6):
        super().__init__()
        self.seq_len = seq_len
        self.normalized = normalized
        self.eps = eps
        self.frequency_filter = nn.Parameter(0.02 * torch.randn(1, seq_len))

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError(
                f"SpectralEntropyEstimator expects [B, C, T], got {tuple(x.shape)}"
            )
        if x.shape[-1] != self.seq_len:
            raise ValueError(
                f"Expected sequence length {self.seq_len}, got {x.shape[-1]}"
            )

        # Remove the DC component, as in the supplied implementation.
        x = x - x.mean(dim=-1, keepdim=True)

        x_fft = torch.fft.rfft(x, dim=-1, norm="ortho")
        filter_fft = torch.fft.rfft(
            self.frequency_filter, dim=-1, norm="ortho"
        )
        filtered_fft = x_fft * filter_fft

        power = filtered_fft.real.square() + filtered_fft.imag.square()
        psd = power / (power.sum(dim=-1, keepdim=True) + self.eps)
        entropy = -(psd * torch.log2(psd + self.eps)).sum(dim=-1)

        if self.normalized:
            n_bins = psd.shape[-1]
            max_entropy = math.log2(max(n_bins, 2))
            entropy = entropy / max_entropy

        return entropy.clamp(0.0, 1.0)


class TransformerEncoderLayer(nn.Module):
    """Encoder layer returning its attention map when available."""

    def __init__(
        self,
        attention,
        d_model,
        d_ff,
        dropout=0.1,
        activation="gelu",
    ):
        super().__init__()
        self.attention = attention
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu if activation == "gelu" else F.relu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x, attn_mask=attn_mask, tau=None, delta=None
        )
        x = self.norm1(x + self.dropout(new_x))

        y = self.dropout(self.activation(self.conv1(x.transpose(1, 2))))
        y = self.dropout(self.conv2(y).transpose(1, 2))
        x = self.norm2(x + y)
        return x, attn


class TransformerEncoder(nn.Module):
    """Small local encoder wrapper that consistently returns attention maps."""

    def __init__(self, layers, norm_layer=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attentions = []
        for layer in self.layers:
            x, attn = layer(x, attn_mask=attn_mask)
            if attn is not None:
                attentions.append(attn)
        if self.norm is not None:
            x = self.norm(x)
        return x, attentions


class PatchTSTBranch(nn.Module):
    """PatchTST-style temporal branch.

    Input/output layout is [B, T, C] / [B, pred_len, C].
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.use_norm = getattr(configs, "use_norm", True)
        self.patch_len = getattr(configs, "patch_len", 16)
        self.stride = getattr(configs, "stride", 8)
        self.d_model = configs.d_model
        self.dropout = getattr(configs, "dropout", 0.1)

        if self.patch_len <= 0 or self.stride <= 0:
            raise ValueError("patch_len and stride must be positive")
        if self.seq_len < self.patch_len:
            raise ValueError(
                f"seq_len ({self.seq_len}) must be >= patch_len ({self.patch_len})"
            )

        # Replication padding at the right end matches the usual PatchTST setup.
        self.patch_num = (self.seq_len - self.patch_len) // self.stride + 2
        self.patch_value_embedding = nn.Linear(self.patch_len, self.d_model)
        self.position_embedding = PositionalEmbedding(self.d_model)
        self.patch_dropout = nn.Dropout(self.dropout)

        d_ff = getattr(configs, "d_ff", 4 * self.d_model)
        n_heads = configs.n_heads
        factor = getattr(configs, "factor", 1)
        activation = getattr(configs, "activation", "gelu")
        e_layers = getattr(configs, "e_layers", 2)

        self.encoder = TransformerEncoder(
            [
                TransformerEncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            factor,
                            attention_dropout=self.dropout,
                            output_attention=False,
                        ),
                        self.d_model,
                        n_heads,
                    ),
                    self.d_model,
                    d_ff,
                    dropout=self.dropout,
                    activation=activation,
                )
                for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(self.d_model),
        )

        self.head = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(self.d_model * self.patch_num, self.pred_len),
            nn.Dropout(self.dropout),
        )

    def forward(self, x):
        # x: [B, T, C]
        batch_size, seq_len, n_vars = x.shape
        if seq_len != self.seq_len:
            raise ValueError(
                f"PatchTSTBranch expected T={self.seq_len}, received T={seq_len}"
            )
        if n_vars != self.enc_in:
            raise ValueError(
                f"PatchTSTBranch expected C={self.enc_in}, received C={n_vars}"
            )

        means, stdev = None, None
        if self.use_norm:
            means = x.mean(dim=1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(
                torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            x = x / stdev

        # [B, T, C] -> [B, C, T]
        x = x.transpose(1, 2)
        x = F.pad(x, (0, self.stride), mode="replicate")
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # [B, C, N, patch_len] -> [B*C, N, patch_len]
        patches = patches.reshape(
            batch_size * n_vars, patches.shape[-2], self.patch_len
        )

        x = self.patch_value_embedding(patches)
        x = x + self.position_embedding(x)
        x = self.patch_dropout(x)
        x, _ = self.encoder(x)

        # [B*C, N, D] -> [B, C, D, N]
        x = x.reshape(batch_size, n_vars, self.patch_num, self.d_model)
        x = x.permute(0, 1, 3, 2)
        pred = self.head(x).transpose(1, 2)  # [B, pred_len, C]

        if self.use_norm:
            pred = pred * stdev[:, 0, :].unsqueeze(1)
            pred = pred + means[:, 0, :].unsqueeze(1)
        return pred


class ITransformerBranch(nn.Module):
    """iTransformer-style variable-token branch with attention-map output."""

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.use_norm = getattr(configs, "use_norm", True)
        self.d_model = configs.d_model
        self.dropout = getattr(configs, "dropout", 0.1)

        self.enc_embedding = DataEmbedding_inverted(
            self.seq_len,
            self.d_model,
            getattr(configs, "embed", "timeF"),
            getattr(configs, "freq", "h"),
            self.dropout,
        )

        d_ff = getattr(configs, "d_ff", 4 * self.d_model)
        n_heads = configs.n_heads
        factor = getattr(configs, "factor", 1)
        activation = getattr(configs, "activation", "gelu")
        e_layers = getattr(configs, "e_layers", 2)

        self.encoder = TransformerEncoder(
            [
                TransformerEncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            factor,
                            attention_dropout=self.dropout,
                            output_attention=True,
                        ),
                        self.d_model,
                        n_heads,
                    ),
                    self.d_model,
                    d_ff,
                    dropout=self.dropout,
                    activation=activation,
                )
                for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(self.d_model),
        )
        self.projector = nn.Linear(self.d_model, self.pred_len, bias=True)

    def forward(self, x, x_mark_enc=None):
        # x: [B, T, C]
        batch_size, seq_len, n_vars = x.shape
        if seq_len != self.seq_len:
            raise ValueError(
                f"ITransformerBranch expected T={self.seq_len}, received T={seq_len}"
            )
        if n_vars != self.enc_in:
            raise ValueError(
                f"ITransformerBranch expected C={self.enc_in}, received C={n_vars}"
            )

        means, stdev = None, None
        if self.use_norm:
            means = x.mean(dim=1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(
                torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            x = x / stdev

        enc_out = self.enc_embedding(x, x_mark_enc)
        enc_out, attentions = self.encoder(enc_out, attn_mask=None)
        pred = self.projector(enc_out).transpose(1, 2)  # [B, pred_len, tokens]

        # DataEmbedding_inverted may append time-feature tokens. Keep only the
        # original variable tokens as forecast channels.
        pred = pred[:, :, :n_vars]

        if self.use_norm:
            pred = pred * stdev[:, 0, :].unsqueeze(1)
            pred = pred + means[:, 0, :].unsqueeze(1)
        return pred, attentions


class EntropyFusionGate(nn.Module):
    """One sigmoid fusion weight per sample, conditioned on two entropies."""

    def __init__(self, hidden_dim=16, dropout=0.0):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        # Start close to an even fusion (g ~= 0.5), while retaining gradients.
        nn.init.xavier_uniform_(self.network[-1].weight, gain=0.1)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, spectral_entropy, attention_entropy):
        # Both inputs are [B]. The output is [B, 1].
        features = torch.stack([spectral_entropy, attention_entropy], dim=-1)
        return torch.sigmoid(self.network(features))


class Model(nn.Module):
    """Entropy-based Adaptive Dual-Transformer (EADT).

    The model has two parallel forecasting branches:
      1. PatchTSTBranch: temporal / patch-wise representation.
      2. ITransformerBranch: variable-token representation.

    A sample-level gate is calculated from:
      - mean normalized spectral entropy across input variables, and
      - normalized attention entropy from iTransformer variable attention only.

    Fusion:
        pred = g * pred_iTransformer + (1 - g) * pred_PatchTST

    `forward` returns only the forecast by default, matching the usual
    time-series forecasting experiment interface. Set `return_aux=True` to
    additionally receive the gate and entropy values for analysis.
    """

    def __init__(self, configs):
        super().__init__()
        self.task_name = getattr(configs, "task_name", "long_term_forecast")
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.use_norm = getattr(configs, "use_norm", True)

        self.patchtst = PatchTSTBranch(configs)
        self.itransformer = ITransformerBranch(configs)
        self.spectral_entropy = SpectralEntropyEstimator(
            self.seq_len, normalized=True
        )
        self.fusion_gate = EntropyFusionGate(
            hidden_dim=getattr(configs, "gate_hidden_dim", 16),
            dropout=getattr(configs, "gate_dropout", 0.0),
        )

        # Detached diagnostics; these do not retain the training computation graph.
        self.last_gate = None
        self.last_spectral_entropy = None
        self.last_attention_entropy = None

    @staticmethod
    def _attention_entropy(attentions, n_vars, eps=1e-8):
        """Compute one normalized attention-entropy value per sample.

        Each attention tensor is expected to have shape [B, H, L, S].
        iTransformer may append time-feature tokens after the C variable tokens.
        This function therefore extracts the C-by-C variable block, row-normalizes
        it, and calculates entropy over keys. Layers, heads, and variable queries
        are averaged, yielding [B].

        Entropy near 0 indicates concentrated attention; entropy near 1 indicates
        diffuse attention. It is a descriptive statistic, not a direct measure of
        predictive usefulness.
        """
        if n_vars <= 1:
            if attentions:
                return attentions[0].new_zeros(attentions[0].shape[0])
            raise RuntimeError("Cannot infer batch size from empty attention list")

        layer_entropies = []
        for attn in attentions:
            if attn is None or attn.ndim != 4:
                continue
            if attn.shape[-2] < n_vars or attn.shape[-1] < n_vars:
                continue

            # Keep only variable-to-variable attention, excluding appended marks.
            variable_attn = attn[:, :, :n_vars, :n_vars]
            row_sum = variable_attn.sum(dim=-1, keepdim=True)
            probs = variable_attn / row_sum.clamp_min(eps)

            entropy = -(
                probs * torch.log(probs.clamp_min(eps))
            ).sum(dim=-1)
            entropy = entropy / math.log(n_vars)

            # Ignore query rows with effectively no attention mass on variables.
            valid_rows = (row_sum.squeeze(-1) > eps).to(entropy.dtype)
            numerator = (entropy * valid_rows).sum(dim=(1, 2))
            denominator = valid_rows.sum(dim=(1, 2)).clamp_min(1.0)
            layer_entropies.append(numerator / denominator)

        if not layer_entropies:
            raise RuntimeError(
                "No usable iTransformer attention maps were returned. "
                "Check that FullAttention(output_attention=True) returns [B,H,L,S]."
            )
        return torch.stack(layer_entropies, dim=0).mean(dim=0).clamp(0.0, 1.0)

    def forecast(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                 return_aux=False):
        if x_enc.ndim != 3:
            raise ValueError(
                f"EADT expects x_enc [B, T, C], got {tuple(x_enc.shape)}"
            )
        if x_enc.shape[1] != self.seq_len or x_enc.shape[2] != self.enc_in:
            raise ValueError(
                "EADT input shape mismatch: expected "
                f"[B, {self.seq_len}, {self.enc_in}], got {tuple(x_enc.shape)}"
            )

        # Both branches receive the same input and are trained end-to-end.
        pred_patch = self.patchtst(x_enc)
        pred_itrans, attentions = self.itransformer(x_enc, x_mark_enc)

        # Use the same input scale used by the branches. Spectral entropy is
        # scale-invariant after power-spectrum normalization; centering is inside
        # SpectralEntropyEstimator.
        spectral_by_var = self.spectral_entropy(x_enc.transpose(1, 2))
        spectral_entropy = spectral_by_var.mean(dim=1)  # [B]
        attention_entropy = self._attention_entropy(attentions, self.enc_in)

        gate = self.fusion_gate(spectral_entropy, attention_entropy)  # [B, 1]
        pred = (
            gate.unsqueeze(1) * pred_itrans
            + (1.0 - gate.unsqueeze(1)) * pred_patch
        )

        self.last_gate = gate.detach()
        self.last_spectral_entropy = spectral_entropy.detach()
        self.last_attention_entropy = attention_entropy.detach()

        if return_aux:
            aux = {
                "gate_itransformer": gate,
                "gate_patchtst": 1.0 - gate,
                "spectral_entropy": spectral_entropy,
                "attention_entropy": attention_entropy,
                "pred_patchtst": pred_patch,
                "pred_itransformer": pred_itrans,
            }
            return pred, aux
        return pred

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
        mask=None,
        return_aux=False,
    ):
        if self.task_name not in ("long_term_forecast", "short_term_forecast"):
            raise NotImplementedError(
                f"EADT currently supports forecasting tasks, got {self.task_name}"
            )
        return self.forecast(
            x_enc,
            x_mark_enc=x_mark_enc,
            x_dec=x_dec,
            x_mark_dec=x_mark_dec,
            return_aux=return_aux,
        )
