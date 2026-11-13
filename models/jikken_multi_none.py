import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Autoformer_EncDec import series_decomp
from layers.Embed import DataEmbedding_wo_pos, DataEmbedding_inverted
from layers.SelfAttention_Family import FullAttention, AttentionLayer


# -----------------------------------------------------------------------------
# TimeMixer-style MDM blocks
# -----------------------------------------------------------------------------
class DFT_series_decomp(nn.Module):
    def __init__(self, top_k=5):
        super().__init__()
        self.top_k = top_k

    def forward(self, x):
        xf = torch.fft.rfft(x, dim=1)
        freq = torch.abs(xf)
        freq[:, 0, :] = 0
        k = min(self.top_k, max(1, freq.shape[1] - 1))
        top_k_freq, _ = torch.topk(freq, k=k, dim=1)
        threshold = top_k_freq[:, -1:, :]
        xf = xf.masked_fill(freq < threshold, 0)
        x_season = torch.fft.irfft(xf, n=x.shape[1], dim=1)
        x_trend = x - x_season
        return x_season, x_trend


class MultiScaleSeasonMixing(nn.Module):
    """Bottom-up mixing of seasonal components (high resolution -> low)."""
    def __init__(self, configs):
        super().__init__()
        self.down_sampling_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(
                    configs.seq_len // (configs.down_sampling_window ** i),
                    configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                ),
                nn.GELU(),
                nn.Linear(
                    configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                ),
            )
            for i in range(configs.down_sampling_layers)
        ])

    def forward(self, season_list):
        out_high = season_list[0]
        out_low = season_list[1]
        out_list = [out_high.permute(0, 2, 1)]
        for i in range(len(season_list) - 1):
            out_low = out_low + self.down_sampling_layers[i](out_high)
            out_high = out_low
            if i + 2 < len(season_list):
                out_low = season_list[i + 2]
            out_list.append(out_high.permute(0, 2, 1))
        return out_list


class MultiScaleTrendMixing(nn.Module):
    """Top-down mixing of trend components (low resolution -> high)."""
    def __init__(self, configs):
        super().__init__()
        self.up_sampling_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(
                    configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    configs.seq_len // (configs.down_sampling_window ** i),
                ),
                nn.GELU(),
                nn.Linear(
                    configs.seq_len // (configs.down_sampling_window ** i),
                    configs.seq_len // (configs.down_sampling_window ** i),
                ),
            )
            for i in reversed(range(configs.down_sampling_layers))
        ])

    def forward(self, trend_list):
        trend_reverse = list(reversed(trend_list))
        out_low = trend_reverse[0]
        out_high = trend_reverse[1]
        out_list = [out_low.permute(0, 2, 1)]
        for i in range(len(trend_reverse) - 1):
            out_high = out_high + self.up_sampling_layers[i](out_low)
            out_low = out_high
            if i + 2 < len(trend_reverse):
                out_high = trend_reverse[i + 2]
            out_list.append(out_low.permute(0, 2, 1))
        out_list.reverse()
        return out_list


class PastDecomposableMixing(nn.Module):
    """MDM block adapted from the supplied TimeMixer implementation."""
    def __init__(self, configs):
        super().__init__()
        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)

        if configs.decomp_method == "moving_avg":
            self.decomposition = series_decomp(configs.moving_avg)
        elif configs.decomp_method == "dft_decomp":
            self.decomposition = DFT_series_decomp(configs.top_k)
        else:
            raise ValueError(
                "decomp_method must be 'moving_avg' or 'dft_decomp', "
                f"got {configs.decomp_method}"
            )

        self.mixing_multi_scale_season = MultiScaleSeasonMixing(configs)
        self.mixing_multi_scale_trend = MultiScaleTrendMixing(configs)
        self.out_cross_layer = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_ff),
            nn.GELU(),
            nn.Linear(configs.d_ff, configs.d_model),
        )

    def forward(self, x_list):
        lengths = [x.shape[1] for x in x_list]
        season_list, trend_list = [], []
        for x in x_list:
            season, trend = self.decomposition(x)
            season_list.append(season.permute(0, 2, 1))
            trend_list.append(trend.permute(0, 2, 1))

        out_season = self.mixing_multi_scale_season(season_list)
        out_trend = self.mixing_multi_scale_trend(trend_list)

        outputs = []
        for original, season, trend, length in zip(
            x_list, out_season, out_trend, lengths
        ):
            mixed = season + trend
            mixed = original + self.out_cross_layer(mixed)
            outputs.append(mixed[:, :length, :])
        return outputs


# -----------------------------------------------------------------------------
# Multi-scale spectral entropy evaluator
# -----------------------------------------------------------------------------
class MultiScaleSpectralEntropy(nn.Module):
    """Computes per-variable spectral entropy and learns scale weights.

    The scale order returned by this module is coarse-to-fine, e.g. [8, 4, 2, 1].
    """
    def __init__(self, seq_len, down_sampling_layers=3,
                 down_sampling_window=2, eps=1e-6, filter_scale=0.02):
        super().__init__()
        self.seq_len = seq_len
        self.eps = eps
        self.scale_factors = [
            down_sampling_window ** i
            for i in range(down_sampling_layers, -1, -1)
        ]
        for factor in self.scale_factors:
            if seq_len // factor < 2:
                raise ValueError(
                    f"seq_len={seq_len} is too short for scale factor {factor}"
                )

        self.frequency_filters = nn.ParameterList([
            nn.Parameter(filter_scale * torch.randn(1, seq_len // factor))
            for factor in self.scale_factors
        ])
        n_scales = len(self.scale_factors)
        hidden = max(8, 2 * n_scales)
        self.scale_selector = nn.Sequential(
            nn.Linear(n_scales, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_scales),
        )

        self.last_scale_weights = None
        self.last_scale_entropies = None

    def _entropy(self, x, frequency_filter):
        x = x - x.mean(dim=-1, keepdim=True)
        xf = torch.fft.rfft(x, dim=-1, norm="ortho")
        ff = torch.fft.rfft(frequency_filter, dim=-1, norm="ortho")
        filtered = xf * ff
        power = filtered.real.square() + filtered.imag.square()
        psd = power / (power.sum(dim=-1, keepdim=True) + self.eps)
        entropy = -(psd * torch.log2(psd + self.eps)).sum(dim=-1)
        max_entropy = torch.log2(
            torch.tensor(psd.shape[-1], dtype=psd.dtype, device=psd.device)
        ).clamp_min(self.eps)
        return (entropy / max_entropy).clamp(0.0, 1.0)

    def forward(self, x):
        # x: [B, C, L]
        if x.ndim != 3 or x.shape[-1] != self.seq_len:
            raise ValueError(
                f"Expected [B,C,{self.seq_len}], got {tuple(x.shape)}"
            )
        scale_entropies = []
        for factor, freq_filter in zip(self.scale_factors, self.frequency_filters):
            scale_x = x if factor == 1 else F.avg_pool1d(
                x, kernel_size=factor, stride=factor
            )
            scale_entropies.append(self._entropy(scale_x, freq_filter))

        # [B, C, n_scales], coarse-to-fine
        entropy_tensor = torch.stack(scale_entropies, dim=-1)
        weights = torch.softmax(self.scale_selector(entropy_tensor), dim=-1)
        self.last_scale_weights = weights.detach()
        self.last_scale_entropies = entropy_tensor.detach()
        return weights, entropy_tensor


# -----------------------------------------------------------------------------
# iTransformer-style variable-wise encoder
# -----------------------------------------------------------------------------
class VariableEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, factor=3):
        super().__init__()
        self.attention = AttentionLayer(
            FullAttention(
                False, factor, attention_dropout=dropout,
                output_attention=False
            ),
            d_model,
            n_heads,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        attended = self.attention(x, x, x, attn_mask=None)[0]
        x = self.norm1(x + self.dropout(attended))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


# -----------------------------------------------------------------------------
# Main model
# -----------------------------------------------------------------------------
class Model(nn.Module):
    """TimeMixer-MDM temporal branch + iTransformer variable branch + SpEn.

    Expected input: x_enc [B, seq_len, enc_in].
    Output: [B, pred_len, c_out] (the training framework may slice it further).
    """
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.features = configs.features
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        self.d_model = configs.d_model
        self.use_norm = bool(getattr(configs, "use_norm", 1))

        # Defaults follow the supplied TimeMixer code where possible.
        self.down_sampling_window = getattr(configs, "down_sampling_window", 2)
        self.down_sampling_layers = getattr(configs, "down_sampling_layers", 3)
        if self.down_sampling_layers < 1:
            raise ValueError("down_sampling_layers must be at least 1")

        # This adapted temporal MDM is channel-independent: each variable gets
        # its own temporal representation; cross-variable learning is handled
        # explicitly by the iTransformer-style branch below.
        mdm_configs = configs
        mdm_configs.down_sampling_window = self.down_sampling_window
        mdm_configs.down_sampling_layers = self.down_sampling_layers
        mdm_configs.decomp_method = getattr(configs, "decomp_method", "moving_avg")
        mdm_configs.moving_avg = getattr(configs, "moving_avg", 25)
        mdm_configs.top_k = getattr(configs, "top_k", 5)

        self.scale_lengths = [
            self.seq_len // (self.down_sampling_window ** i)
            for i in range(self.down_sampling_layers + 1)
        ]
        if any(length < 2 for length in self.scale_lengths):
            raise ValueError(
                "The selected seq_len/down_sampling_layers produce a scale "
                f"shorter than 2 points: {self.scale_lengths}"
            )

        # Temporal branch: scale-wise embeddings + MDM blocks.
        self.temporal_embedding = DataEmbedding_wo_pos(
            1, configs.d_model, configs.embed, configs.freq, configs.dropout
        )
        self.mdm_blocks = nn.ModuleList([
            PastDecomposableMixing(mdm_configs)
            for _ in range(configs.e_layers)
        ])
        self.temporal_predict_layers = nn.ModuleList([
            nn.Linear(length, self.pred_len)
            for length in self.scale_lengths
        ])
        self.temporal_projection = nn.Linear(configs.d_model, 1, bias=True)

        # Variable branch: inverted embedding and self-attention over variables.
        self.variable_embedding = DataEmbedding_inverted(
            self.seq_len, configs.d_model, configs.embed,
            configs.freq, configs.dropout
        )
        self.variable_encoder = nn.ModuleList([
            VariableEncoderLayer(
                configs.d_model,
                configs.n_heads,
                configs.d_ff,
                configs.dropout,
                getattr(configs, "factor", 3),
            )
            for _ in range(getattr(configs, "variable_e_layers", configs.e_layers))
        ])
        self.variable_prediction = nn.Linear(configs.d_model, self.pred_len)

        # Spectral-entropy-based adaptive scale selector.
        self.use_spectral_entropy = getattr(
            configs, "use_dependency_evaluator", False
        )
        self.spectral_entropy = (
            MultiScaleSpectralEntropy(
                seq_len=self.seq_len,
                down_sampling_layers=self.down_sampling_layers,
                down_sampling_window=self.down_sampling_window,
            )
            if self.use_spectral_entropy else None
        )

        # Learned fusion between the two directional branches.
        self.branch_fusion = nn.Linear(2 * self.pred_len, 1)

    def _normalize(self, x):
        means, stdev = None, None
        if self.use_norm:
            means = x.mean(dim=1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(
                torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5
            ).detach()
            x = x / stdev
        return x, means, stdev

    def _make_multiscale_inputs(self, x):
        # x: [B, L, C] -> list of [B, L_i, C], fine-to-coarse.
        x_list = [x]
        current = x.permute(0, 2, 1)
        for _ in range(self.down_sampling_layers):
            current = F.avg_pool1d(
                current,
                kernel_size=self.down_sampling_window,
                stride=self.down_sampling_window,
            )
            x_list.append(current.permute(0, 2, 1))
        return x_list

    def _make_multiscale_marks(self, x_mark_enc):
        if x_mark_enc is None:
            return None
        marks = [x_mark_enc]
        current = x_mark_enc
        for _ in range(self.down_sampling_layers):
            current = current[:, ::self.down_sampling_window, :]
            marks.append(current)
        return marks

    def _temporal_branch(self, x, x_mark_enc, scale_weights):
        # x: [B,L,C]. Each variable is treated as an independent temporal series.
        B, _, C = x.shape
        x_scales = self._make_multiscale_inputs(x)
        mark_scales = self._make_multiscale_marks(x_mark_enc)

        embedded_scales = []
        for i, x_scale in enumerate(x_scales):
            _, length, _ = x_scale.shape
            # [B,L_i,C] -> [B*C,L_i,1]
            x_individual = x_scale.permute(0, 2, 1).contiguous().reshape(
                B * C, length, 1
            )
            if mark_scales is not None:
                mark = mark_scales[i]
                # Match the actual pooled sequence length in case of floor division.
                mark = mark[:, :length, :]
                mark = mark.repeat_interleave(C, dim=0)
            else:
                mark = None
            embedded_scales.append(self.temporal_embedding(x_individual, mark))

        for block in self.mdm_blocks:
            embedded_scales = block(embedded_scales)

        # Each scale independently produces a forecast. TimeMixer scales are
        # fine-to-coarse; SpEn returns coarse-to-fine, so reverse its weights.
        scale_predictions = []
        for i, (encoded, predict_layer) in enumerate(
            zip(embedded_scales, self.temporal_predict_layers)
        ):
            # [B*C,L_i,D] -> [B*C,pred_len,D] -> [B*C,pred_len,1]
            aligned = predict_layer(encoded.transpose(1, 2)).transpose(1, 2)
            pred = self.temporal_projection(aligned).squeeze(-1)
            pred = pred.reshape(B, C, self.pred_len).permute(0, 2, 1)
            scale_predictions.append(pred)

        predictions = torch.stack(scale_predictions, dim=-1)  # [B,P,C,S]
        if scale_weights is None:
            weights = torch.full(
                (B, C, len(scale_predictions)),
                1.0 / len(scale_predictions),
                device=x.device,
                dtype=x.dtype,
            )
        else:
            weights = scale_weights.flip(-1)  # coarse-to-fine -> fine-to-coarse
        weighted = predictions * weights.unsqueeze(1)
        temporal_forecast = weighted.sum(dim=-1)
        return temporal_forecast

    def _variable_branch(self, x, x_mark_enc):
        # Inverted embedding: [B,L,C] -> [B,C,D] (plus any calendar tokens).
        variable_tokens = self.variable_embedding(x, None)
        variable_tokens = variable_tokens[:, :self.enc_in, :]
        for layer in self.variable_encoder:
            variable_tokens = layer(variable_tokens)
        # [B,C,D] -> [B,C,P] -> [B,P,C]
        return self.variable_prediction(variable_tokens).transpose(1, 2)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x, means, stdev = self._normalize(x_enc)

        if self.spectral_entropy is not None:
            scale_weights, _ = self.spectral_entropy(x.permute(0, 2, 1))
        else:
            scale_weights = None

        temporal_pred = self._temporal_branch(x, x_mark_enc, scale_weights)
        variable_pred = self._variable_branch(x, x_mark_enc)

        # One learned fusion coefficient per sample and variable.
        fusion_input = torch.cat(
            [temporal_pred.transpose(1, 2), variable_pred.transpose(1, 2)],
            dim=-1,
        )
        alpha = torch.sigmoid(self.branch_fusion(fusion_input))
        output = (
            alpha * temporal_pred.transpose(1, 2)
            + (1.0 - alpha) * variable_pred.transpose(1, 2)
        ).transpose(1, 2)

        if self.use_norm:
            output = output * stdev[:, 0, :].unsqueeze(1)
            output = output + means[:, 0, :].unsqueeze(1)

        if self.features == "MS":
            output = output[:, :, -1:]
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ("long_term_forecast", "short_term_forecast"):
            return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        raise NotImplementedError(
            "This adapted model currently implements long-/short-term forecasting only."
        )
