import torch

from models.base_model import BaseModel
from .backbone import DGCNN
from .rpn import RPN
from .memory_encoder import MemoryEncoder
from .transformer import Transformer


class ChronoTrack(BaseModel):
    def __init__(self, cfg, log):
        super().__init__(cfg, log)
        self.cfg = cfg
        self.backbone = DGCNN(cfg.backbone_cfg)
        self.transformer = Transformer(cfg.transformer_cfg)
        self.loc_net = RPN(cfg.rpn_cfg)
        self.memory_encoder = MemoryEncoder(cfg.memory_encoder_cfg)

    def forward_embed(self, input):
        pcds = input["pcds"]
        batch_size, duration, npts, _ = pcds.shape

        pcds = pcds.view(batch_size * duration, npts, -1)
        b_output = self.backbone(pcds)
        xyz = b_output["xyz"]
        feat = b_output["feat"]
        idx = b_output["idx"]
        assert len(idx.shape) == 2
        return dict(
            xyzs=xyz.view(batch_size, duration, xyz.shape[1], xyz.shape[2]),  # b,t,n,3
            feats=feat.view(batch_size, duration, feat.shape[1], feat.shape[2]),  # b,t,c,n
            idxs=idx.view(batch_size, duration, idx.shape[1]),  # b,t,n
        )

    def forward_update(self, input):
        memory = input.pop("memory", None)
        geo_feats = input["geo_feats"]  # b, c, n
        xyz = input["xyz"]  # b, n, 3
        mask = input["mask"]  # b, n

        prev_fg_tokens = memory["fg_tokens"] if memory is not None else None
        fg_tokens = self.memory_encoder(xyz, geo_feats, mask, prev_fg_tokens)

        new_memory = dict()
        # b,n_token,c
        new_memory["fg_tokens"] = fg_tokens
        # b,c,t,n
        new_memory["feat"] = torch.cat((memory["feat"], geo_feats.unsqueeze(2)), dim=2) if memory is not None else geo_feats.unsqueeze(2)
        # b,t,n,3
        new_memory["xyz"] = torch.cat((memory["xyz"], xyz.unsqueeze(1)), dim=1) if memory is not None else xyz.unsqueeze(1)
        # b,t,n
        new_memory["mask"] = torch.cat((memory["mask"], mask.unsqueeze(1)), dim=1) if memory is not None else mask.unsqueeze(1)

        if new_memory["feat"].shape[2] > self.cfg.bg_memory_size:
            new_memory["feat"] = new_memory["feat"][:, :, 1:, :]
            new_memory["xyz"] = new_memory["xyz"][:, 1:, :, :]
            new_memory["mask"] = new_memory["mask"][:, 1:, :]

        return dict(memory=new_memory)

    def forward_localize(self, input):
        return self.loc_net(input)

    def forward_propagate(self, input):
        memory = input.pop("memory", None)
        feat = input.pop("feat")
        xyz = input.pop("xyz")

        assert memory is not None, "Memory is None!"

        trfm_input = dict(memory=memory, feat=feat, xyz=xyz)
        trfm_output = self.transformer(trfm_input)

        return trfm_output

    def forward(self, input, mode):
        forward_dict = dict(
            embed=self.forward_embed,
            propagate=self.forward_propagate,
            localize=self.forward_localize,
            update=self.forward_update,
        )
        assert mode in forward_dict, "%s has not been supported" % mode

        forward_func = forward_dict[mode]
        output = forward_func(input)
        return output
