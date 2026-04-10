"""
GRIT with structure-rotated attention AND an AlphaFold2-style pair-update
edge stream.

Identical to `GritRoPETransformerLayer` in every respect EXCEPT how the
edge features are updated layer-to-layer.  Where the RoPE-only variant
kept GRIT's original

    e_ij^{l+1} = e_ij^l + O_e( signed_sqrt((K_j + Q_i) * E_w) + E_b )

this variant replaces the inner update with a symmetric residual MLP

    e_ij^{l+1} = e_ij^l + MLP( LN([h_i + h_j || h_i * h_j || e_ij^l]) )

that is shared across heads within a layer and (optionally) across all
layers of the network. The symmetric concat guarantees that e_ij and e_ji
evolve identically, which is the right inductive bias for undirected
molecular graphs like ZINC.

Config surface (under cfg.gt.attn.edge_mlp):

    hidden_mult          : int  (default 2)   -- MLP is 3d -> hidden_mult*d -> d
    dropout              : float (default 0.0)
    act                  : str  (default 'gelu')  'gelu' | 'relu'
    share_across_layers  : bool (default False)  if True, a single MLP is
                                                  reused by every layer

Select with:

    model:
      type: GritRoPEPairTransformer
    gt:
      layer_type: GritRoPEPairTransformer
      attn:
        edge_mlp:
          hidden_mult: 2          # try 1 for no expansion (3d -> d -> d)
          share_across_layers: False
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
from grit.layer.grit_rope_layer import MultiHeadAttentionLayerGritRoPE


# --------------------------------------------------------------------------
# The pair-update MLP
# --------------------------------------------------------------------------

class SymmetricPairEdgeMLP(nn.Module):
    """Symmetric residual pair update for undirected edges.

    Input features per edge (i,j):
        [ h_i + h_j   ||   h_i * h_j   ||   e_ij ]       shape (E, 3d)

    The symmetric sum and product guarantee invariance under swapping i<->j,
    so e_ij and e_ji evolve identically without the model having to learn
    that equivariance from data.

    Architecture (pre-LayerNorm, GeLU by default):
        LN -> Linear(3d, h*d) -> act -> Dropout -> Linear(h*d, d)

    The caller adds the residual (e_new = e_old + self(h_i, h_j, e_old)).

    Output projection is zero-initialised so the module starts as an exact
    identity update -- handy when bolting onto an already-working attention
    baseline.
    """

    def __init__(self, d_model: int, hidden_mult: int = 2,
                 dropout: float = 0.0, act: str = "gelu"):
        super().__init__()
        self.d_model = d_model
        self.hidden_mult = hidden_mult

        in_dim = 3 * d_model
        hid_dim = hidden_mult * d_model

        self.norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, d_model)
        self.drop = nn.Dropout(dropout)

        if act == "gelu":
            self.act = nn.GELU()
        elif act == "relu":
            self.act = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation {act!r} for SymmetricPairEdgeMLP")

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        # Zero-init the output projection so the residual starts as identity.
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, h_src: torch.Tensor, h_dst: torch.Tensor,
                e: torch.Tensor) -> torch.Tensor:
        """All three inputs are (num_edges, d_model). Returns the delta."""
        pair = torch.cat([h_src + h_dst, h_src * h_dst, e], dim=-1)
        pair = self.norm(pair)
        pair = self.fc1(pair)
        pair = self.act(pair)
        pair = self.drop(pair)
        pair = self.fc2(pair)
        return pair


# --------------------------------------------------------------------------
# The outer Transformer layer
# --------------------------------------------------------------------------

@register_layer("GritRoPEPairTransformer")
class GritRoPEPairTransformerLayer(nn.Module):
    """GRIT layer: RoPE dot-product attention + symmetric pair-MLP edges."""

    def __init__(self, in_dim, out_dim, num_heads,
                 dropout=0.0,
                 attn_dropout=0.0,
                 layer_norm=False, batch_norm=True,
                 residual=True,
                 act='relu',
                 norm_e=True,
                 O_e=True,
                 shared_edge_mlp=None,
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

        # --- Attention: same as RoPE variant, but with its internal edge
        #     update DISABLED. We own the edge update out here. ---
        self.attention = MultiHeadAttentionLayerGritRoPE(
            in_dim=in_dim,
            out_dim=out_dim // num_heads,
            num_heads=num_heads,
            use_bias=cfg.attn.get("use_bias", False),
            dropout=attn_dropout,
            clamp=cfg.attn.get("clamp", 5.),
            compute_edge_update=False,
            cfg=cfg,
        )

        self.O_h = nn.Linear(out_dim // num_heads * num_heads, out_dim)

        # --- The pair-MLP that replaces GRIT's edge-stream update. ---
        edge_mlp_cfg = cfg.attn.get("edge_mlp", CN())
        hidden_mult = int(edge_mlp_cfg.get("hidden_mult", 2))
        edge_mlp_dropout = float(edge_mlp_cfg.get("dropout", 0.0))
        edge_mlp_act = edge_mlp_cfg.get("act", "gelu")

        if shared_edge_mlp is not None:
            # Cross-layer sharing path: the network passes us one pre-built
            # instance to reuse. Do NOT register as a child module, otherwise
            # parameters get double-counted by torch.nn (and saved twice in
            # the checkpoint). Using object.__setattr__ sidesteps nn.Module's
            # child registration.
            object.__setattr__(self, "edge_mlp", shared_edge_mlp)
            self._owns_edge_mlp = False
        else:
            self.edge_mlp = SymmetricPairEdgeMLP(
                d_model=out_dim,
                hidden_mult=hidden_mult,
                dropout=edge_mlp_dropout,
                act=edge_mlp_act,
            )
            self._owns_edge_mlp = True

        # --- Degree scaler (carried over from GRIT) ---
        if self.deg_scaler:
            self.deg_coef = nn.Parameter(
                torch.zeros(1, out_dim // num_heads * num_heads, 2)
            )
            nn.init.xavier_normal_(self.deg_coef)

        # --- Norms around the attention / edge update residuals ---
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

        # --- Node-stream FFN (identical to GritTransformerLayer) ---
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
        num_nodes = batch.num_nodes
        log_deg = get_log_deg(batch)

        h_in1 = batch.x                                   # (N, d_model)
        e_in1 = batch.get("edge_attr", None)              # (num_edges, d_model) or None

        # ---- Attention (node update only; no edge update inside) ----
        h_attn_out, _unused = self.attention(batch)
        h = h_attn_out.view(num_nodes, -1)
        h = F.dropout(h, self.dropout, training=self.training)

        if self.deg_scaler:
            h = torch.stack([h, h * log_deg], dim=-1)
            h = (h * self.deg_coef).sum(dim=-1)

        h = self.O_h(h)

        # ---- Pair-MLP edge update ----
        # We use the PRE-attention h (`h_in1`) and PRE-update e (`e_in1`) so
        # this matches the formula e^{l+1} = e^l + f(h^l, h^l, e^l) literally.
        # The pair MLP is zero-initialised, so at training step 0 this is an
        # exact identity (e_new == e_in1) and the model recovers baseline
        # behaviour.
        if e_in1 is not None:
            src = batch.edge_index[0]
            dst = batch.edge_index[1]
            h_src = h_in1[src]                            # (num_edges, d_model)
            h_dst = h_in1[dst]                            # (num_edges, d_model)
            delta_e = self.edge_mlp(h_src, h_dst, e_in1)  # (num_edges, d_model)
            e = e_in1 + delta_e
        else:
            e = None

        # ---- Residual + norms around the attention block ----
        if self.residual:
            if self.rezero:
                h = h * self.alpha1_h
            h = h_in1 + h
            # e already includes its own residual via (e_in1 + delta_e).
            # If rezero is on, apply the scale to the delta component only.
            if e is not None and self.rezero:
                e = e_in1 + self.alpha1_e * (e - e_in1)

        if self.layer_norm:
            h = self.layer_norm1_h(h)
            if e is not None:
                e = self.layer_norm1_e(e)
        if self.batch_norm:
            h = self.batch_norm1_h(h)
            if e is not None:
                e = self.batch_norm1_e(e)

        # ---- Node FFN (same as GRIT) ----
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
        shared = "shared" if not self._owns_edge_mlp else "per-layer"
        return (
            '{}(in_channels={}, out_channels={}, heads={}, '
            'd_sem={}, d_struct={}, edge_mlp={}, residual={})'
        ).format(
            self.__class__.__name__,
            self.in_channels, self.out_channels, self.num_heads,
            self.attention.d_sem, self.attention.d_struct,
            shared, self.residual,
        )


# --------------------------------------------------------------------------
# Network class: mostly mirrors GritTransformer, but optionally builds ONE
# shared SymmetricPairEdgeMLP and passes it to every layer.
# --------------------------------------------------------------------------

class _FeatureEncoder(nn.Module):
    """Same as the FeatureEncoder in grit/network/grit_model.py.

    Duplicated here so this file doesn't depend on importing an internal
    private symbol from the network module.
    """

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


@register_network("GritRoPEPairTransformer")
class GritRoPEPairTransformer(nn.Module):
    """GRIT architecture with RoPE attention + AF2-style pair-MLP edges."""

    def __init__(self, dim_in, dim_out):
        super().__init__()
        cfg = global_cfg

        self.encoder = _FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        if cfg.posenc_RRWP.enable:
            self.rrwp_abs_encoder = register.node_encoder_dict["rrwp_linear"](
                cfg.posenc_RRWP.ksteps, cfg.gnn.dim_inner)
            rel_pe_dim = cfg.posenc_RRWP.ksteps
            self.rrwp_rel_encoder = register.edge_encoder_dict["rrwp_linear"](
                rel_pe_dim, cfg.gnn.dim_edge,
                pad_to_full_graph=cfg.gt.attn.full_attn,
                add_node_attr_as_self_loop=False,
                fill_value=0.,
            )

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(dim_in, cfg.gnn.dim_inner,
                                   cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        assert cfg.gt.dim_hidden == cfg.gnn.dim_inner == dim_in, \
            "The inner and hidden dims must match."

        # --- Decide on edge-MLP sharing policy ---
        edge_mlp_cfg = cfg.gt.attn.get("edge_mlp", CN())
        share_across_layers = bool(edge_mlp_cfg.get("share_across_layers", False))
        hidden_mult = int(edge_mlp_cfg.get("hidden_mult", 2))
        edge_mlp_dropout = float(edge_mlp_cfg.get("dropout", 0.0))
        edge_mlp_act = edge_mlp_cfg.get("act", "gelu")

        shared_edge_mlp = None
        if share_across_layers:
            shared_edge_mlp = SymmetricPairEdgeMLP(
                d_model=cfg.gt.dim_hidden,
                hidden_mult=hidden_mult,
                dropout=edge_mlp_dropout,
                act=edge_mlp_act,
            )
            # Register as a direct child so its parameters are counted once
            # and saved once in the state dict, regardless of how many layers
            # reference it.
            self.shared_edge_mlp = shared_edge_mlp

        TransformerLayer = register.layer_dict.get("GritRoPEPairTransformer")
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
                shared_edge_mlp=shared_edge_mlp,
                cfg=cfg.gt,
            ))
        self.layers = nn.Sequential(*layers)

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch
