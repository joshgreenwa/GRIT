"""
GRIT with structure-rotated attention (Option A).

Replaces GRIT's element-wise additive attention with standard scaled
dot-product attention. Each head's content dimension is split into a
semantic channel (plain dot product) and a structural channel whose
Q/K are rotated RoPE-style, where the rotation angles are a learned
linear function of each node's RRWP diagonal (its "structural footprint").

  z_ij = (q_sem . k_sem) / sqrt(d_sem)
       + (q_str . k_str) / sqrt(d_struct)
       + e_ij
  alpha_ij = softmax_j(z_ij)
  h_i^{l+1} = h_i + W_O sum_j alpha_ij V h_j

GRIT's edge stream is preserved: the per-channel edge features are updated
via the same (K + Q) * E_w -> signed_sqrt -> + E_b pathway that GRIT uses,
so E^{l+1} flows to the next layer the way it always did in GRIT.

Dropped relative to GRIT:
  - Aw           (no learned reduction; logit comes from the dot product)
  - VeRow        (edge-enhance side term; not part of the clean overview)

The outer Transformer layer (residual, FFN, norm, degree scaler) is the
same as the original GritTransformerLayer.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
from torch_scatter import scatter

from torch_geometric.graphgym.config import cfg as global_cfg
from torch_geometric.graphgym.register import register_layer, act_dict

from yacs.config import CfgNode as CN

from grit.layer.grit_layer import pyg_softmax, get_log_deg


def _apply_rotation(x, cos_a, sin_a):
    """RoPE-style 2D rotation on the last dim.

    x:     (..., d_struct)             paired as (x0,x1),(x2,x3),...
    cos_a: (..., d_struct // 2)
    sin_a: (..., d_struct // 2)
    """
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rot_even = x_even * cos_a - x_odd * sin_a
    rot_odd = x_even * sin_a + x_odd * cos_a
    out = torch.stack([rot_even, rot_odd], dim=-1)
    return out.flatten(-2)


class MultiHeadAttentionLayerGritRoPE(nn.Module):
    """Structure-rotated dot-product attention with GRIT edge stream."""

    def __init__(self, in_dim, out_dim, num_heads, use_bias,
                 clamp=5., dropout=0.,
                 d_struct_ratio=0.75,
                 ksteps=None,
                 cfg=CN(),
                 **kwargs):
        super().__init__()

        # NOTE: `out_dim` here is the per-head dimension (d_head), matching
        # how GritTransformerLayer constructs MultiHeadAttentionLayerGritSparse.
        self.d_head = out_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.clamp = np.abs(clamp) if clamp is not None else None

        # --- Resolve d_sem / d_struct (d_struct must be even) ---
        # Priority: explicit cfg.attn.d_struct  >  d_struct_ratio  >  default 3/4.
        d_struct_cfg = cfg.attn.get("d_struct", None) if hasattr(cfg, "attn") else None
        if d_struct_cfg is not None:
            d_struct = int(d_struct_cfg)
        else:
            ratio = cfg.attn.get("d_struct_ratio", d_struct_ratio) \
                if hasattr(cfg, "attn") else d_struct_ratio
            d_struct = int(round(ratio * self.d_head))
        if d_struct % 2 != 0:
            d_struct -= 1
        d_struct = max(0, min(self.d_head, d_struct))
        d_sem = self.d_head - d_struct
        assert d_struct % 2 == 0
        assert d_sem > 0 or d_struct > 0
        self.d_sem = d_sem
        self.d_struct = d_struct
        self.num_rot = d_struct // 2

        # --- Q / K / V / E projections ---
        self.Q = nn.Linear(in_dim, self.d_head * num_heads, bias=True)
        self.K = nn.Linear(in_dim, self.d_head * num_heads, bias=use_bias)
        self.V = nn.Linear(in_dim, self.d_head * num_heads, bias=use_bias)
        # GRIT edge projection, kept verbatim: feeds E_w / E_b for edge update.
        self.E = nn.Linear(in_dim, self.d_head * num_heads * 2, bias=True)
        # Scalar edge bias w_e: E_ij -> one scalar per head.
        self.w_e = nn.Linear(in_dim, num_heads, bias=True)

        nn.init.xavier_normal_(self.Q.weight)
        nn.init.xavier_normal_(self.K.weight)
        nn.init.xavier_normal_(self.V.weight)
        nn.init.xavier_normal_(self.E.weight)
        nn.init.xavier_normal_(self.w_e.weight)

        # --- Structural rotation weights W_angles, per head ---
        # Shape: (num_heads, K, d_struct // 2). K = RRWP walk length.
        if ksteps is None:
            ksteps = int(global_cfg.posenc_RRWP.ksteps)
        self.ksteps = ksteps
        if self.num_rot > 0:
            self.W_angles = nn.Parameter(
                torch.empty(num_heads, ksteps, self.num_rot)
            )
            # Small init: angles start near zero so rotation ~ identity.
            nn.init.normal_(self.W_angles, mean=0.0, std=0.02)
        else:
            self.register_parameter("W_angles", None)

    # -----------------------------------------------------------------

    def _compute_angles(self, rrwp_diag):
        """rrwp_diag: (N, K)  ->  angles: (N, H, d_struct/2)."""
        # einsum: for each node n, head h, rotation r:
        #   angles[n,h,r] = sum_k rrwp_diag[n,k] * W_angles[h,k,r]
        return torch.einsum("nk,hkr->nhr", rrwp_diag, self.W_angles)

    def propagate_attention(self, batch):
        edge_index = batch.edge_index
        src_idx, dst_idx = edge_index[0], edge_index[1]

        K_src = batch.K_h[src_idx]   # (E, H, d_head)
        Q_dst = batch.Q_h[dst_idx]   # (E, H, d_head)

        # --- Standard scaled dot-product logit, split by channel ---
        if self.d_sem > 0 and self.d_struct > 0:
            sem_logit = (Q_dst[..., :self.d_sem] * K_src[..., :self.d_sem]).sum(-1) \
                / math.sqrt(self.d_sem)
            str_logit = (Q_dst[..., self.d_sem:] * K_src[..., self.d_sem:]).sum(-1) \
                / math.sqrt(self.d_struct)
            logit = sem_logit + str_logit
        elif self.d_struct > 0:
            logit = (Q_dst * K_src).sum(-1) / math.sqrt(self.d_struct)
        else:
            logit = (Q_dst * K_src).sum(-1) / math.sqrt(self.d_sem)
        # logit: (E, H)

        # --- Scalar edge bias from the edge stream ---
        if batch.get("e_bias", None) is not None:
            logit = logit + batch.e_bias  # (E, H)

        if self.clamp is not None:
            logit = torch.clamp(logit, min=-self.clamp, max=self.clamp)

        # --- Edge stream update (GRIT's f, unchanged) ---
        # Produces updated per-channel edge features for the next layer.
        if batch.get("E", None) is not None:
            E_proj = batch.E.view(-1, self.num_heads, self.d_head * 2)
            E_w = E_proj[..., :self.d_head]
            E_b = E_proj[..., self.d_head:]
            e_t = (K_src + Q_dst) * E_w
            e_t = torch.sqrt(torch.relu(e_t)) - torch.sqrt(torch.relu(-e_t))
            e_t = e_t + E_b
            batch.wE = e_t.flatten(1)

        # --- Softmax over incoming edges per destination node ---
        # pyg_softmax handles arbitrary trailing dims via scatter_* on dim 0.
        attn = pyg_softmax(logit.unsqueeze(-1), dst_idx)  # (E, H, 1)
        attn = self.dropout(attn)
        batch.attn = attn

        # --- Value aggregation ---
        msg = batch.V_h[src_idx] * attn  # (E, H, d_head)
        wV = torch.zeros_like(batch.V_h)
        scatter(msg, dst_idx, dim=0, out=wV, reduce='add')
        batch.wV = wV

    def forward(self, batch):
        N = batch.num_nodes
        H, D = self.num_heads, self.d_head

        Q_h = self.Q(batch.x).view(N, H, D)
        K_h = self.K(batch.x).view(N, H, D)
        V_h = self.V(batch.x).view(N, H, D)

        # --- Structural rotation on the d_struct slice of Q and K ---
        if self.d_struct > 0:
            if not hasattr(batch, "rrwp") or batch.rrwp is None:
                raise RuntimeError(
                    "MultiHeadAttentionLayerGritRoPE requires batch.rrwp "
                    "(set by add_full_rrwp / posenc_RRWP.enable=True)."
                )
            rrwp_diag = batch.rrwp  # (N, K)
            assert rrwp_diag.size(-1) == self.ksteps, (
                f"rrwp diag dim {rrwp_diag.size(-1)} != ksteps {self.ksteps}; "
                "did posenc_RRWP.ksteps change since layer init?"
            )
            angles = self._compute_angles(rrwp_diag)  # (N, H, d_struct/2)
            cos_a = torch.cos(angles)
            sin_a = torch.sin(angles)

            if self.d_sem > 0:
                Q_sem, Q_str = Q_h[..., :self.d_sem], Q_h[..., self.d_sem:]
                K_sem, K_str = K_h[..., :self.d_sem], K_h[..., self.d_sem:]
                Q_str = _apply_rotation(Q_str, cos_a, sin_a)
                K_str = _apply_rotation(K_str, cos_a, sin_a)
                Q_h = torch.cat([Q_sem, Q_str], dim=-1)
                K_h = torch.cat([K_sem, K_str], dim=-1)
            else:
                Q_h = _apply_rotation(Q_h, cos_a, sin_a)
                K_h = _apply_rotation(K_h, cos_a, sin_a)

        batch.Q_h = Q_h
        batch.K_h = K_h
        batch.V_h = V_h

        if batch.get("edge_attr", None) is not None:
            batch.E = self.E(batch.edge_attr)
            batch.e_bias = self.w_e(batch.edge_attr)  # (num_edges, H)
        else:
            batch.E = None
            batch.e_bias = None

        self.propagate_attention(batch)

        h_out = batch.wV
        e_out = batch.get("wE", None)
        return h_out, e_out


# ------------------------------------------------------------------------
# Outer Transformer layer — same structure as GritTransformerLayer.
# ------------------------------------------------------------------------

@register_layer("GritRoPETransformer")
class GritRoPETransformerLayer(nn.Module):
    """GRIT Transformer layer using structure-rotated dot-product attention."""

    def __init__(self, in_dim, out_dim, num_heads,
                 dropout=0.0,
                 attn_dropout=0.0,
                 layer_norm=False, batch_norm=True,
                 residual=True,
                 act='relu',
                 norm_e=True,
                 O_e=True,
                 cfg=dict(),
                 **kwargs):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm

        self.update_e = cfg.get("update_e", True)
        self.bn_momentum = cfg.bn_momentum
        self.bn_no_runner = cfg.bn_no_runner
        self.rezero = cfg.get("rezero", False)

        self.act = act_dict[act]() if act is not None else nn.Identity()
        if cfg.get("attn", None) is None:
            cfg.attn = CN()
        self.deg_scaler = cfg.attn.get("deg_scaler", True)

        self.attention = MultiHeadAttentionLayerGritRoPE(
            in_dim=in_dim,
            out_dim=out_dim // num_heads,
            num_heads=num_heads,
            use_bias=cfg.attn.get("use_bias", False),
            dropout=attn_dropout,
            clamp=cfg.attn.get("clamp", 5.),
            cfg=cfg,
        )

        self.O_h = nn.Linear(out_dim // num_heads * num_heads, out_dim)
        self.O_e = nn.Linear(out_dim // num_heads * num_heads, out_dim) if O_e else nn.Identity()

        if self.deg_scaler:
            self.deg_coef = nn.Parameter(
                torch.zeros(1, out_dim // num_heads * num_heads, 2)
            )
            nn.init.xavier_normal_(self.deg_coef)

        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_dim)
            self.layer_norm1_e = nn.LayerNorm(out_dim) if norm_e else nn.Identity()

        if self.batch_norm:
            self.batch_norm1_h = nn.BatchNorm1d(
                out_dim, track_running_stats=not self.bn_no_runner,
                eps=1e-5, momentum=cfg.bn_momentum,
            )
            self.batch_norm1_e = nn.BatchNorm1d(
                out_dim, track_running_stats=not self.bn_no_runner,
                eps=1e-5, momentum=cfg.bn_momentum,
            ) if norm_e else nn.Identity()

        # FFN for h
        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_h_layer2 = nn.Linear(out_dim * 2, out_dim)

        if self.layer_norm:
            self.layer_norm2_h = nn.LayerNorm(out_dim)
        if self.batch_norm:
            self.batch_norm2_h = nn.BatchNorm1d(
                out_dim, track_running_stats=not self.bn_no_runner,
                eps=1e-5, momentum=cfg.bn_momentum,
            )

        if self.rezero:
            self.alpha1_h = nn.Parameter(torch.zeros(1, 1))
            self.alpha2_h = nn.Parameter(torch.zeros(1, 1))
            self.alpha1_e = nn.Parameter(torch.zeros(1, 1))

    def forward(self, batch):
        h = batch.x
        num_nodes = batch.num_nodes
        log_deg = get_log_deg(batch)

        h_in1 = h
        e_in1 = batch.get("edge_attr", None)
        e = None

        h_attn_out, e_attn_out = self.attention(batch)

        h = h_attn_out.view(num_nodes, -1)
        h = F.dropout(h, self.dropout, training=self.training)

        if self.deg_scaler:
            h = torch.stack([h, h * log_deg], dim=-1)
            h = (h * self.deg_coef).sum(dim=-1)

        h = self.O_h(h)
        if e_attn_out is not None:
            e = e_attn_out.flatten(1)
            e = F.dropout(e, self.dropout, training=self.training)
            e = self.O_e(e)

        if self.residual:
            if self.rezero:
                h = h * self.alpha1_h
            h = h_in1 + h
            if e is not None:
                if self.rezero:
                    e = e * self.alpha1_e
                e = e + e_in1

        if self.layer_norm:
            h = self.layer_norm1_h(h)
            if e is not None:
                e = self.layer_norm1_e(e)

        if self.batch_norm:
            h = self.batch_norm1_h(h)
            if e is not None:
                e = self.batch_norm1_e(e)

        h_in2 = h
        h = self.FFN_h_layer1(h)
        h = self.act(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)

        if self.residual:
            if self.rezero:
                h = h * self.alpha2_h
            h = h_in2 + h

        if self.layer_norm:
            h = self.layer_norm2_h(h)
        if self.batch_norm:
            h = self.batch_norm2_h(h)

        batch.x = h
        if self.update_e:
            batch.edge_attr = e
        else:
            batch.edge_attr = e_in1

        return batch

    def __repr__(self):
        return (
            '{}(in_channels={}, out_channels={}, heads={}, '
            'd_sem={}, d_struct={}, residual={})'
        ).format(
            self.__class__.__name__,
            self.in_channels, self.out_channels, self.num_heads,
            self.attention.d_sem, self.attention.d_struct, self.residual,
        )
