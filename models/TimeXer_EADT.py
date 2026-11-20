import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted, PositionalEmbedding


# -----------------------------------------------------------------------------
# Utility: normalized entropy of variable-to-variable cross-attention
# -----------------------------------------------------------------------------
def normalized_attention_entropy(attn, n_key_vars, eps=1e-8):
    """Calculate one normalized cross-attention entropy per sample.

    Args:
        attn: Attention map with shape [B, H, Q, K].
              In TimeXer, the first n_key_vars keys correspond to variables;
              any appended time-feature tokens are excluded.
        n_key_vars: Number of variable tokens among the keys.
        eps: Numerical-stability constant.

    Returns:
        Tensor with shape [B], approximately in [0, 1].

    The entropy is averaged over attention heads and query tokens. Rows with
    effectively zero attention mass on variable keys are ignored.
    """
    if attn is None:
        raise RuntimeError(
            "Cross-attention weights were not returned. Set "
            "output_attention=True for the TimeXer cross-attention layers."
        )
    if attn.ndim != 4:
        raise ValueError(
            "Expected cross-attention map [B, H, Q, K], "
            f"but received shape {tuple(attn.shape)}."
        )

    batch_size = attn.shape[0]
    if n_key_vars <= 1:
        # Entropy over one or zero variable keys is defined as zero.
        return attn.new_zeros(batch_size)
    if attn.shape[-1] < n_key_vars:
        raise ValueError(
            f"Attention map has {attn.shape[-1]} keys, but "
            f"n_key_vars={n_key_vars} was requested."
        )

    # Keep variable keys only; exclude appended time-feature tokens.
    variable_attn = attn[..., :n_key_vars]
    row_sum = variable_attn.sum(dim=-1, keepdim=True)
    probs = variable_attn / row_sum.clamp_min(eps)

    entropy = -(
        probs * torch.log(probs.clamp_min(eps))
    ).sum(dim=-1)
    entropy = entropy / torch.log(
        attn.new_tensor(float(n_key_vars))
    ).clamp_min(eps)

    # Ignore query rows with effectively no attention mass on variable keys.
    valid_rows = (row_sum.squeeze(-1) > eps).to(entropy.dtype)
    numerator = (entropy * valid_rows).sum(dim=(1, 2))
    denominator = valid_rows.sum(dim=(1, 2)).clamp_min(1.0)
    return (numerator / denominator).clamp(0.0, 1.0)


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # [B, n_vars, d_model, patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        return self.dropout(x)


class EnEmbedding(nn.Module):
    def __init__(self, n_vars, d_model, patch_len, dropout):
        super().__init__()
        self.patch_len = patch_len
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.glb_token = nn.Parameter(torch.randn(1, n_vars, 1, d_model))
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, n_vars, seq_len]
        n_vars = x.shape[1]
        glb = self.glb_token.repeat((x.shape[0], 1, 1, 1))

        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        # [B, n_vars, patch_num, patch_len] -> [B*n_vars, patch_num, patch_len]
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        x = self.value_embedding(x) + self.position_embedding(x)
        x = x.reshape(-1, n_vars, x.shape[-2], x.shape[-1])
        x = torch.cat([x, glb], dim=2)
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        return self.dropout(x), n_vars


class DependencyEvaluator(nn.Module):
    """SEED-style spectral-entropy evaluator.

    Input : x in [B, n_vars, seq_len]
    Output: normalized spectral entropy in [B, n_vars], approximately [0, 1].

    A learnable time-domain filter is transformed to the frequency domain and
    applied to the input spectrum before calculating spectral entropy.
    """

    def __init__(self, seq_len, normalized=True, eps=1e-6):
        super().__init__()
        self.normalized = normalized
        self.eps = eps
        self.filter_scale = 0.02
        self.frequency_filter = nn.Parameter(
            self.filter_scale * torch.randn(1, seq_len)
        )

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError(
                f"DependencyEvaluator expects [B, n_vars, seq_len], got {tuple(x.shape)}"
            )
        if x.shape[-1] != self.frequency_filter.shape[-1]:
            raise ValueError(
                "Input sequence length does not match the configured seq_len: "
                f"got {x.shape[-1]}, expected {self.frequency_filter.shape[-1]}"
            )

        # Remove the DC component before examining the spectrum.
        x = x - x.mean(dim=-1, keepdim=True)

        # Learnable spectral shaping filter.
        x_fft = torch.fft.rfft(x, dim=-1, norm="ortho")
        filter_fft = torch.fft.rfft(
            self.frequency_filter, dim=-1, norm="ortho"
        )
        filtered_fft = x_fft * filter_fft

        power = filtered_fft.real.square() + filtered_fft.imag.square()
        psd = power / (power.sum(dim=-1, keepdim=True) + self.eps)
        entropy = -(psd * torch.log2(psd + self.eps)).sum(dim=-1)

        if self.normalized:
            num_bins = psd.shape[-1]
            max_entropy = torch.log2(
                torch.tensor(num_bins, dtype=psd.dtype, device=psd.device)
            ).clamp_min(self.eps)
            entropy = entropy / max_entropy

        return entropy.clamp(0.0, 1.0)


class EntropyFusionGate(nn.Module):
    """Produce one sample-level gate from spectral and attention entropy."""

    def __init__(self, hidden_dim=16, dropout=0.0):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        # Initialize near an even gate while keeping the gate trainable.
        nn.init.xavier_uniform_(self.network[-1].weight, gain=0.1)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, spectral_entropy, attention_entropy):
        # Both inputs are [B]. The output is [B, 1].
        features = torch.stack(
            [spectral_entropy, attention_entropy], dim=-1
        )
        return torch.sigmoid(self.network(features))


class EncoderLayer(nn.Module):
    def __init__(
        self,
        self_attention,
        cross_attention,
        d_model,
        d_ff=None,
        dropout=0.1,
        activation="relu",
    ):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(
        self,
        x,
        cross,
        x_mask=None,
        cross_mask=None,
        tau=None,
        delta=None,
        dependency_weight=None,
        spectral_entropy_mean=None,
        previous_attention_entropies=None,
        entropy_fusion_gate=None,
        n_key_vars=None,
        apply_attention_entropy_gate=False,
    ):
        # x: [B*n_vars, patch_num+1, D]
        # cross: [B, n_exogenous_tokens, D]
        batch_size, _, d_model = cross.shape
        if x.shape[0] % batch_size != 0:
            raise ValueError(
                "TimeXer shape mismatch: x batch dimension must be divisible by "
                f"cross batch dimension ({x.shape[0]} vs {batch_size})."
            )
        n_series = x.shape[0] // batch_size

        # 1) Temporal/patch-wise self-attention.
        # Attention weights are not needed from this branch.
        x = x + self.dropout(
            self.self_attention(
                x, x, x, attn_mask=x_mask, tau=tau, delta=None
            )[0]
        )
        x = self.norm1(x)

        # 2) Cross-attention for the global token of each endogenous series.
        x_glb_ori = x[:, -1, :].reshape(batch_size, n_series, d_model)
        x_glb_attn, cross_attn = self.cross_attention(
            x_glb_ori,
            cross,
            cross,
            attn_mask=cross_mask,
            tau=tau,
            delta=delta,
        )

        current_attention_entropy = None
        attention_gate = None
        combined_attention_entropy = None

        if n_key_vars is not None:
            current_attention_entropy = normalized_attention_entropy(
                cross_attn, n_key_vars
            )

        # Preserve the existing variable-wise SpEn weighting.
        if dependency_weight is not None:
            if dependency_weight.shape != (batch_size, n_series):
                raise ValueError(
                    "dependency_weight must have shape [B, n_series]; "
                    f"got {tuple(dependency_weight.shape)}, expected "
                    f"{(batch_size, n_series)}."
                )
            spectral_gate = dependency_weight.to(
                device=x_glb_attn.device, dtype=x_glb_attn.dtype
            ).unsqueeze(-1)
            x_glb_attn = x_glb_attn * spectral_gate

        # Use the mean entropy across all Cross-Attention layers to obtain
        # exactly one additional gate per sample. It is applied at the final
        # Encoder layer, where the entropies from all layers are available.
        if apply_attention_entropy_gate:
            if current_attention_entropy is None:
                raise RuntimeError(
                    "Attention-entropy gating was requested, but no usable "
                    "Cross-Attention map was returned."
                )
            if entropy_fusion_gate is None:
                raise RuntimeError(
                    "entropy_fusion_gate must be provided when "
                    "apply_attention_entropy_gate=True."
                )

            entropy_list = list(previous_attention_entropies or [])
            entropy_list.append(current_attention_entropy)
            combined_attention_entropy = torch.stack(
                entropy_list, dim=0
            ).mean(dim=0)

            if spectral_entropy_mean is None:
                spectral_entropy_mean = x_glb_attn.new_zeros(batch_size)
            attention_gate = entropy_fusion_gate(
                spectral_entropy_mean, combined_attention_entropy
            )  # [B, 1]
            x_glb_attn = x_glb_attn * attention_gate.unsqueeze(-1)

        x_glb = x_glb_ori + self.dropout(x_glb_attn)
        x_glb = self.norm2(x_glb)
        x_glb = x_glb.reshape(batch_size * n_series, 1, d_model)

        # 3) Feed-forward block.
        y = torch.cat([x[:, :-1, :], x_glb], dim=1)
        y_ff = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y_ff = self.dropout(self.conv2(y_ff).transpose(-1, 1))
        x = self.norm3(y + y_ff)

        return x, current_attention_entropy, attention_gate, combined_attention_entropy


class Encoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(
        self,
        x,
        cross,
        x_mask=None,
        cross_mask=None,
        tau=None,
        delta=None,
        dependency_weight=None,
        spectral_entropy_mean=None,
        entropy_fusion_gate=None,
        n_key_vars=None,
        use_attention_entropy=True,
    ):
        attention_entropies = []
        attention_gate = None
        combined_attention_entropy = None

        for layer_index, layer in enumerate(self.layers):
            is_last_layer = layer_index == len(self.layers) - 1
            apply_gate_here = use_attention_entropy and is_last_layer

            (
                x,
                layer_attention_entropy,
                layer_gate,
                layer_combined_entropy,
            ) = layer(
                x,
                cross,
                x_mask=x_mask,
                cross_mask=cross_mask,
                tau=tau,
                delta=delta,
                dependency_weight=dependency_weight,
                spectral_entropy_mean=spectral_entropy_mean,
                previous_attention_entropies=attention_entropies,
                entropy_fusion_gate=entropy_fusion_gate,
                n_key_vars=n_key_vars if use_attention_entropy else None,
                apply_attention_entropy_gate=apply_gate_here,
            )

            if layer_attention_entropy is not None:
                attention_entropies.append(layer_attention_entropy)
            if layer_gate is not None:
                attention_gate = layer_gate
                combined_attention_entropy = layer_combined_entropy

        if self.norm is not None:
            x = self.norm(x)
        if self.projection is not None:
            x = self.projection(x)

        if attention_entropies:
            # This is the same layer-averaged entropy used to calculate the gate.
            attention_entropy = torch.stack(
                attention_entropies, dim=0
            ).mean(dim=0)
        else:
            attention_entropy = None

        diagnostics = {
            "attention_entropy": attention_entropy,
            "attention_gate": attention_gate,
            "layer_attention_entropies": attention_entropies,
            "combined_attention_entropy": combined_attention_entropy,
        }
        return x, diagnostics


class Model(nn.Module):
    """TimeXer with spectral-entropy and attention-entropy gating.

    - Spectral entropy provides the original variable-wise SpEn weights.
    - Cross-Attention entropy is calculated from variable keys only; appended
      time-feature tokens are excluded.
    - The mean Cross-Attention entropy across Encoder layers is one scalar per
      sample. An MLP combines it with mean spectral entropy to produce one
      sample-level gate.
    - The additional sample-level gate is applied to the Cross-Attention output
      in the final Encoder layer. Earlier layers retain the original SpEn gate.

    Set configs.use_attention_entropy=False for a SpEn-only ablation.
    Set configs.use_dependency_evaluator=False to disable SpEn weighting. If
    attention-entropy gating remains enabled in that case, its spectral-entropy
    input to the fusion MLP is set to zero.
    """

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.features = configs.features
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.use_norm = configs.use_norm
        self.patch_len = configs.patch_len
        self.patch_num = int(configs.seq_len // configs.patch_len)
        self.n_vars = 1 if configs.features == "MS" else configs.enc_in

        self.use_attention_entropy = getattr(
            configs, "use_attention_entropy", True
        )
        self.use_dependency_evaluator = getattr(
            configs, "use_dependency_evaluator", True
        )

        self.en_embedding = EnEmbedding(
            self.n_vars, configs.d_model, self.patch_len, configs.dropout
        )
        self.ex_embedding = DataEmbedding_inverted(
            configs.seq_len,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
        )

        self.dependency_evaluator = (
            DependencyEvaluator(configs.seq_len, normalized=True)
            if self.use_dependency_evaluator
            else None
        )
        self.entropy_fusion_gate = (
            EntropyFusionGate(
                hidden_dim=getattr(configs, "gate_hidden_dim", 16),
                dropout=getattr(configs, "gate_dropout", 0.0),
            )
            if self.use_attention_entropy
            else None
        )

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=False,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=self.use_attention_entropy,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
        )

        self.head_nf = configs.d_model * (self.patch_num + 1)
        self.head = FlattenHead(
            configs.enc_in,
            self.head_nf,
            configs.pred_len,
            head_dropout=configs.dropout,
        )

        # Detached diagnostics for logging/analysis. They do not retain graphs.
        self.last_gate = None
        self.last_spectral_entropy = None
        self.last_spectral_entropy_by_var = None
        self.last_attention_entropy = None
        self.last_layer_attention_entropies = None

    def _normalize(self, x_enc):
        """Apply TimeXer's per-series normalization, returning stats for denorm."""
        means, stdev = None, None
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            x_enc = x_enc / stdev
        return x_enc, means, stdev

    def _encode(self, en_embed, ex_embed, x_enc, multi_series):
        batch_size = x_enc.shape[0]

        if self.dependency_evaluator is not None:
            # [B, T, C] -> [B, C, T]
            spectral_entropy_by_var = self.dependency_evaluator(
                x_enc.permute(0, 2, 1)
            )
            spectral_entropy_mean = spectral_entropy_by_var.mean(dim=1)
            if multi_series:
                dependency_weight = spectral_entropy_by_var
            else:
                # The endogenous branch contains only the last target variable.
                dependency_weight = spectral_entropy_by_var[:, -1:]
        else:
            spectral_entropy_by_var = None
            spectral_entropy_mean = x_enc.new_zeros(batch_size)
            dependency_weight = None

        if multi_series:
            # Cross keys begin with all original variable tokens.
            n_key_vars = self.enc_in
        else:
            # In MS mode, the target is endogenous and the other variables
            # provide Cross-Attention context.
            n_key_vars = max(self.enc_in - 1, 0)

        enc_out, diagnostics = self.encoder(
            en_embed,
            ex_embed,
            dependency_weight=dependency_weight,
            spectral_entropy_mean=spectral_entropy_mean,
            entropy_fusion_gate=self.entropy_fusion_gate,
            n_key_vars=n_key_vars,
            use_attention_entropy=self.use_attention_entropy,
        )

        self.last_spectral_entropy = spectral_entropy_mean.detach()
        self.last_spectral_entropy_by_var = (
            spectral_entropy_by_var.detach()
            if spectral_entropy_by_var is not None
            else None
        )
        self.last_attention_entropy = (
            diagnostics["attention_entropy"].detach()
            if diagnostics["attention_entropy"] is not None
            else None
        )
        self.last_gate = (
            diagnostics["attention_gate"].detach()
            if diagnostics["attention_gate"] is not None
            else None
        )
        self.last_layer_attention_entropies = [
            entropy.detach()
            for entropy in diagnostics["layer_attention_entropies"]
        ]

        return enc_out

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x_enc, means, stdev = self._normalize(x_enc)

        # MS/S setup: the last variable is endogenous; the others are exogenous.
        en_embed, n_vars = self.en_embedding(
            x_enc[:, :, -1].unsqueeze(-1).permute(0, 2, 1)
        )
        ex_embed = self.ex_embedding(x_enc[:, :, :-1], x_mark_enc)

        enc_out = self._encode(
            en_embed, ex_embed, x_enc, multi_series=False
        )
        enc_out = enc_out.reshape(
            -1, n_vars, enc_out.shape[-2], enc_out.shape[-1]
        )
        enc_out = enc_out.permute(0, 1, 3, 2)

        dec_out = self.head(enc_out).permute(0, 2, 1)

        if self.use_norm:
            dec_out = dec_out * stdev[:, 0, -1:].unsqueeze(1).repeat(
                1, self.pred_len, 1
            )
            dec_out = dec_out + means[:, 0, -1:].unsqueeze(1).repeat(
                1, self.pred_len, 1
            )
        return dec_out

    def forecast_multi(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x_enc, means, stdev = self._normalize(x_enc)

        en_embed, n_vars = self.en_embedding(x_enc.permute(0, 2, 1))
        ex_embed = self.ex_embedding(x_enc, x_mark_enc)

        enc_out = self._encode(
            en_embed, ex_embed, x_enc, multi_series=True
        )
        enc_out = enc_out.reshape(
            -1, n_vars, enc_out.shape[-2], enc_out.shape[-1]
        )
        enc_out = enc_out.permute(0, 1, 3, 2)

        dec_out = self.head(enc_out).permute(0, 2, 1)

        if self.use_norm:
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(
                1, self.pred_len, 1
            )
            dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(
                1, self.pred_len, 1
            )
        return dec_out

    def forward(
        self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None
    ):
        if self.task_name in (
            "long_term_forecast",
            "short_term_forecast",
        ):
            if self.features == "M":
                dec_out = self.forecast_multi(
                    x_enc, x_mark_enc, x_dec, x_mark_dec
                )
            else:
                dec_out = self.forecast(
                    x_enc, x_mark_enc, x_dec, x_mark_dec
                )
            return dec_out[:, -self.pred_len :, :]
        return None
