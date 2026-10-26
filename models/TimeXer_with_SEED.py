import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted, PositionalEmbedding


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
    """SEED-style spectral entropy evaluator.

    Input : x in [B, n_vars, seq_len]
    Output: normalized spectral entropy in [B, n_vars], approximately [0, 1].

    As in the supplied SEED implementation, a learnable time-domain filter is
    transformed to the frequency domain and applied to the input spectrum
    before calculating spectral entropy.
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

        # Learnable spectral shaping filter, following the provided SEED code.
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

        # Protect the gate from tiny numerical excursions outside [0, 1].
        return entropy.clamp(0.0, 1.0)


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
    ):
        for layer in self.layers:
            x = layer(
                x,
                cross,
                x_mask=x_mask,
                cross_mask=cross_mask,
                tau=tau,
                delta=delta,
                dependency_weight=dependency_weight,
            )

        if self.norm is not None:
            x = self.norm(x)
        if self.projection is not None:
            x = self.projection(x)
        return x


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
        x = x + self.dropout(
            self.self_attention(
                x, x, x, attn_mask=x_mask, tau=tau, delta=None
            )[0]
        )
        x = self.norm1(x)

        # 2) Cross-attention for the global token of each endogenous series.
        # Group each batch's global tokens to [B, n_series, D] for cross-attention.
        x_glb_ori = x[:, -1, :].reshape(batch_size, n_series, d_model)
        x_glb_attn = self.cross_attention(
            x_glb_ori,
            cross,
            cross,
            attn_mask=cross_mask,
            tau=tau,
            delta=delta,
        )[0]

        # Dependency Evaluator gate:
        # high spectral entropy -> more external/variable-wise context;
        # low spectral entropy -> keep the model closer to its own temporal pattern.
        if dependency_weight is not None:
            if dependency_weight.shape != (batch_size, n_series):
                raise ValueError(
                    "dependency_weight must have shape [B, n_series]; "
                    f"got {tuple(dependency_weight.shape)}, expected "
                    f"{(batch_size, n_series)}."
                )
            gate = dependency_weight.to(
                device=x_glb_attn.device, dtype=x_glb_attn.dtype
            ).unsqueeze(-1)
            x_glb_attn = x_glb_attn * gate

        x_glb = x_glb_ori + self.dropout(x_glb_attn)
        x_glb = self.norm2(x_glb)
        x_glb = x_glb.reshape(batch_size * n_series, 1, d_model)

        # 3) Feed-forward block.
        y = torch.cat([x[:, :-1, :], x_glb], dim=1)
        y_ff = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y_ff = self.dropout(self.conv2(y_ff).transpose(-1, 1))
        return self.norm3(y + y_ff)


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.features = configs.features
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = configs.use_norm
        self.patch_len = configs.patch_len
        self.patch_num = int(configs.seq_len // configs.patch_len)
        self.n_vars = 1 if configs.features == "MS" else configs.enc_in

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
                            output_attention=False,
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

        # Enabled by default. Set configs.use_dependency_evaluator=False for
        # a vanilla-TimeXer ablation without spectral-entropy gating.
        self.use_dependency_evaluator = getattr(
            configs, "use_dependency_evaluator", True
        )
        self.dependency_evaluator = (
            DependencyEvaluator(configs.seq_len, normalized=True)
            if self.use_dependency_evaluator
            else None
        )
        
        # Disable spectral-entropy gating
        #self.use_dependency_evaluator = False
        #self.dependency_evaluator = None

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

    def _dependency_weight(self, x_enc, multi_series):
        if self.dependency_evaluator is None:
            return None

        # x_enc: [B, T, C] -> spectral evaluator expects [B, C, T].
        spectral_entropy = self.dependency_evaluator(x_enc.permute(0, 2, 1))
        if multi_series:
            # features='M': one weight for each variable/global token.
            return spectral_entropy
        # features='MS': TimeXer uses the last input variable as the target.
        # The endogenous patch branch contains only that target variable.
        return spectral_entropy[:, -1:]

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x_enc, means, stdev = self._normalize(x_enc)
        dependency_weight = self._dependency_weight(x_enc, multi_series=False)

        # The last variable is the endogenous/target series; the others provide
        # cross-variable context (the standard TimeXer MS setup).
        en_embed, n_vars = self.en_embedding(
            x_enc[:, :, -1].unsqueeze(-1).permute(0, 2, 1)
        )
        ex_embed = self.ex_embedding(x_enc[:, :, :-1], x_mark_enc)

        enc_out = self.encoder(
            en_embed, ex_embed, dependency_weight=dependency_weight
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
        dependency_weight = self._dependency_weight(x_enc, multi_series=True)

        en_embed, n_vars = self.en_embedding(x_enc.permute(0, 2, 1))
        ex_embed = self.ex_embedding(x_enc, x_mark_enc)

        enc_out = self.encoder(
            en_embed, ex_embed, dependency_weight=dependency_weight
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

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ("long_term_forecast", "short_term_forecast"):
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