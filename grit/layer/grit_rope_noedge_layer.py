"""
GRIT with structure-rotated attention ONLY -- no edge representation.

This is the purest "RoPE-only" ablation: every layer's attention logit is
computed entirely from the dot-product of (optionally rotated) Q and K, with
NO edge-derived bias or edge stream update.  The only structural signal
available to the model is the RoPE rotation, whose angles are learned linear
functions of each node's RRWP diagonal.

  z_ij = (q_sem . k_sem) / sqrt(d_sem)
       + (q_str . k_str) / sqrt(d_struct)
  alpha_ij = softmax_j(z_ij)
  h_i^{l+1} = h_i + W_O sum_j alpha_ij V h_j

No edge features are maintained across layers.  Edge encodings (RRWP
relative) are still computed in the network for full-graph padding but are
ignored by the attention layers.

Select with:

    model:
      type: GritRoPENoEdgeTransformer
    gt:
      layer_type: GritRoPENoEdgeTransformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.graphgym.config import cfg as global_cfg
from torch_geometric.graphgym.models.gnn import GNNPreMP
from torch_geometric.graphgym.models.layer import (new_layer_config,
                                                   BatchNorm1dNode)
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.register import (register_layer,
                                                register_network,
                                                act_dict)
from yacs.config import CfgNode as CN

from grit.layer.grit_layer import get_log_deg
from grit.layer.grit_rope_layer import (MultiHeadAttentionLayerGritRoPE,
                                         StaticRRWPBiasMLP,
                                         StaticRRWPBiasComputer)


# --------------------------------------------------------------------------
# Layer
# --------------------------------------------------------------------------

@register_layer("GritRoPENoEdgeTransformer")
class GritRoPENoEdgeTransformerLayer(nn.Module):
    """GRIT layer with pure RoPE dot-product attention and no edge stream."""

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

        self.bn_momentum = cfg.bn_momentum
        self.bn_no_runner = cfg.bn_no_runner
        self.rezero = cfg.get("rezero", False)

        self.act = act_dict[act]() if act is not None else nn.Identity()
        if cfg.get("attn", None) is None:
            cfg.attn = CN()
        self.deg_scaler = cfg.attn.get("deg_scaler", True)

        # --- Attention: no edge update, no edge bias ---
        self.attention = MultiHeadAttentionLayerGritRoPE(
            in_dim=in_dim,
            out_dim=out_dim // num_heads,
            num_heads=num_heads,
            use_bias=cfg.attn.get("use_bias", False),
            dropout=attn_dropout,
            clamp=cfg.attn.get("clamp", 5.),
            compute_edge_update=False,
            use_edge_bias=False,
            cfg=cfg,
        )

        self.O_h = nn.Linear(out_dim // num_heads * num_heads, out_dim)

        # --- Degree scaler (carried over from GRIT) ---
        if self.deg_scaler:
            self.deg_coef = nn.Parameter(
                torch.zeros(1, out_dim // num_heads * num_heads, 2)
            )
            nn.init.xavier_normal_(self.deg_coef)

        # --- Norms (node only -- no edge norms needed) ---
        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_dim)
        if self.batch_norm:
            self.batch_norm1_h = nn.BatchNorm1d(
                out_dim, track_running_stats=not self.bn_no_runner,
                eps=1e-5, momentum=cfg.bn_momentum,
            )

        # --- Node FFN ---
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

    def forward(self, batch):
        num_nodes = batch.num_nodes
        log_deg = get_log_deg(batch)

        h_in1 = batch.x

        # ---- Attention (node only; no edge interaction) ----
        h_attn_out, _unused = self.attention(batch)
        h = h_attn_out.view(num_nodes, -1)
        h = F.dropout(h, self.dropout, training=self.training)

        if self.deg_scaler:
            h = torch.stack([h, h * log_deg], dim=-1)
            h = (h * self.deg_coef).sum(dim=-1)

        h = self.O_h(h)

        # ---- Residual + norms ----
        if self.residual:
            if self.rezero:
                h = h * self.alpha1_h
            h = h_in1 + h

        if self.layer_norm:
            h = self.layer_norm1_h(h)
        if self.batch_norm:
            h = self.batch_norm1_h(h)

        # ---- Node FFN ----
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
        # Edge attr is untouched -- kept in batch for topology only.
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


# --------------------------------------------------------------------------
# Network
# --------------------------------------------------------------------------

class _FeatureEncoder(nn.Module):
    """Same as FeatureEncoder in grit_model.py (duplicated to avoid
    importing a private symbol from the network module)."""

    def __init__(self, dim_in):
        super().__init__()
        self.dim_in = dim_in
        cfg = global_cfg
        if cfg.dataset.node_encoder:
            NodeEncoder = register.node_encoder_dict[cfg.dataset.node_encoder_name]
            self.node_encoder = NodeEncoder(cfg.gnn.dim_inner)
            if cfg.dataset.node_encoder_bn:
                self.node_encoder_bn = BatchNorm1dNode(
                    new_layer_config(cfg.gnn.dim_inner, -1, -1,
                                     has_act=False, has_bias=False, cfg=cfg))
            self.dim_in = cfg.gnn.dim_inner
        if cfg.dataset.edge_encoder:
            if 'PNA' in cfg.gt.layer_type:
                cfg.gnn.dim_edge = min(128, cfg.gnn.dim_inner)
            else:
                cfg.gnn.dim_edge = cfg.gnn.dim_inner
            EdgeEncoder = register.edge_encoder_dict[cfg.dataset.edge_encoder_name]
            self.edge_encoder = EdgeEncoder(cfg.gnn.dim_edge)
            if cfg.dataset.edge_encoder_bn:
                self.edge_encoder_bn = BatchNorm1dNode(
                    new_layer_config(cfg.gnn.dim_edge, -1, -1,
                                     has_act=False, has_bias=False, cfg=cfg))

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch


@register_network("GritRoPENoEdgeTransformer")
class GritRoPENoEdgeTransformer(nn.Module):
    """GRIT architecture with pure RoPE attention and no edge stream.

    The RRWP relative encoder is still used to pad the graph to a full graph
    (when cfg.gt.attn.full_attn is True) so that all node pairs have edges in
    the edge_index. However, the edge features are never read or updated by
    the transformer layers -- all structural information flows exclusively
    through the RoPE rotation angles derived from each node's RRWP diagonal.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        cfg = global_cfg

        self.encoder = _FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        if cfg.posenc_RRWP.enable:
            # Absolute encoder: feeds batch.rrwp (the node RRWP diagonal)
            # which the RoPE rotation angles are derived from.
            self.rrwp_abs_encoder = register.node_encoder_dict["rrwp_linear"](
                cfg.posenc_RRWP.ksteps, cfg.gnn.dim_inner)
            # Relative encoder: still needed for full-graph padding
            # (pad_to_full_graph). The edge features it produces are ignored
            # by the layers but the edge_index is needed.
            rel_pe_dim = cfg.posenc_RRWP.ksteps
            self.rrwp_rel_encoder = register.edge_encoder_dict["rrwp_linear"](
                rel_pe_dim, cfg.gnn.dim_edge,
                pad_to_full_graph=cfg.gt.attn.full_attn,
                add_node_attr_as_self_loop=False,
                fill_value=0.,
            )

        # --- Optional static RRWP bias MLP ---
        # Particularly useful for the no-edge variant: gives the model pairwise
        # structural info (from the RRWP relative encoding) as a static
        # attention bias, without maintaining a full edge stream.
        rrwp_bias_cfg = cfg.gt.attn.get("rrwp_bias", CN())
        if rrwp_bias_cfg.get("enable", False):
            bias_mlp = StaticRRWPBiasMLP(
                d_edge=cfg.gnn.dim_edge,
                n_heads=cfg.gt.n_heads,
                hidden_mult=int(rrwp_bias_cfg.get("hidden_mult", 2)),
                act=rrwp_bias_cfg.get("act", "relu"),
            )
            self.static_rrwp_bias = StaticRRWPBiasComputer(bias_mlp)

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(dim_in, cfg.gnn.dim_inner,
                                   cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        assert cfg.gt.dim_hidden == cfg.gnn.dim_inner == dim_in, \
            "The inner and hidden dims must match."

        TransformerLayer = register.layer_dict.get("GritRoPENoEdgeTransformer")
        layers = []
        for _ in range(cfg.gt.layers):
            layers.append(TransformerLayer(
                in_dim=cfg.gt.dim_hidden,
                out_dim=cfg.gt.dim_hidden,
                num_heads=cfg.gt.n_heads,
                dropout=cfg.gt.dropout,
                act=cfg.gnn.act,
                attn_dropout=cfg.gt.attn_dropout,
                layer_norm=cfg.gt.layer_norm,
                batch_norm=cfg.gt.batch_norm,
                residual=True,
                norm_e=cfg.gt.attn.norm_e,
                O_e=cfg.gt.attn.O_e,
                cfg=cfg.gt,
            ))
        self.layers = nn.Sequential(*layers)

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch
