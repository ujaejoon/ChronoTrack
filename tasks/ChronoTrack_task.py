import torch.nn.functional as F
import torch

import pytorch3d

from datasets.utils.pcd_utils import *
from .base_task import BaseTask


class ChronoTrackTask(BaseTask):
    def __init__(self, cfg, log):
        super().__init__(cfg, log)

    def build_mask_loss(self, input):
        pred = input["pred"]
        gt = input["gt"]
        return F.binary_cross_entropy_with_logits(pred, gt)

    def build_bbox_loss(self, input):
        pred = input["pred"]
        gt = input["gt"]
        mask = input["mask"]
        loss = F.smooth_l1_loss(pred, gt, reduction="none")
        loss = (loss.mean(2) * mask).sum() / (mask.sum() + 1e-6)
        return loss

    def build_objectness_loss(self, input):
        pred = input["pred"]
        gt = input["gt"]
        mask = input["mask"]
        loss = F.binary_cross_entropy_with_logits(pred, gt, pos_weight=torch.tensor([2.0], device=self.device), reduction="none")
        loss = (loss * mask).sum() / (mask.sum() + 1e-6)
        return loss

    def build_center_loss(self, input):
        pred = input["pred"]
        gt = input["gt"]
        mask = input["mask"]
        loss = F.mse_loss(pred, gt, reduction="none")
        loss = (loss.mean(2) * mask).sum() / (mask.sum() + 1e-06)
        return loss

    @staticmethod
    def normalize_to_canonical(xyz, bbox):
        # xyz: b,n,3 or b,t,n,3
        # bbox: b,4 or b,t,4
        # return: b,n,3 or b,t,n,3

        reshape = False
        if len(xyz.shape) == 4:
            B, T, N, _ = xyz.shape
            xyz = xyz.view(B * T, N, 3)
            bbox = bbox.view(B * T, 4)
            reshape = True

        bbox_center = bbox[:, :3].unsqueeze(1)  # b,1,3
        xyz = xyz - bbox_center  # b,n,3
        bbox_orientation_degree = bbox[:, 3].unsqueeze(1)  # b,1
        bbox_orientation_radian = torch.deg2rad(bbox_orientation_degree)  # b,1
        axis_angle = torch.cat(
            [
                torch.zeros_like(bbox_orientation_radian),  # b,1
                torch.zeros_like(bbox_orientation_radian),  # b,1
                bbox_orientation_radian,  # b,1
            ],
            dim=1,
        )  # b,3
        quaternion = pytorch3d.transforms.axis_angle_to_quaternion(axis_angle).unsqueeze(1)  # b,1,4
        xyz = pytorch3d.transforms.quaternion_apply(quaternion, xyz)  # b,n,3

        if reshape:
            xyz = xyz.view(B, T, N, 3)

        return xyz

    def build_temporal_consistency_loss(self, input):
        geo_feats = input["geo_feats"]  # b,t,c,n
        geo_feats = geo_feats.permute(0, 1, 3, 2)  # b,t,n,c
        mask_gts = input["mask_gts"]  # b,t,n
        bbox_gts = input["bbox_gts"]  # b,t,4
        xyzs = input["xyzs"]  # b,t,n,3

        new_xyzs = self.normalize_to_canonical(xyzs, bbox_gts)  # b,t,n,3

        loss = 0.0
        cnt = 0
        for i in range(0, geo_feats.size(1) - 1):  # 0~t-2

            j_range = range(i + 1, geo_feats.size(1))  # i+1~t-1

            for j in j_range:
                if i == j:
                    continue
                else:
                    cnt += 1

                geo_feat_i = geo_feats[:, i, :, :]  # b,n,c
                geo_feat_j = geo_feats[:, j, :, :]  # b,n,c
                mask_gt_i = mask_gts[:, i, :]
                mask_gt_j = mask_gts[:, j, :]
                new_xyz_i = new_xyzs[:, i, :, :]  # b,n,3
                new_xyz_j = new_xyzs[:, j, :, :]  # b,n,3

                # prev to curr nearest neighbor in canonical space and gather feature
                pairwise_dist = torch.norm(new_xyz_i.unsqueeze(2) - new_xyz_j.unsqueeze(1), dim=-1)
                nn_dist, nn_idx = pairwise_dist.min(dim=2)
                # dist_mask = nn_dist < 0.3
                dist_mask = nn_dist < self.cfg.loss_cfg.temporal_consistency_distance_threshold
                nn_feat = torch.gather(geo_feat_j, 1, nn_idx.unsqueeze(-1).expand(-1, -1, geo_feat_j.size(-1)))
                nn_mask_gt = torch.gather(mask_gt_j, 1, nn_idx)
                pair_mask = ((mask_gt_i > 0.5) * (nn_mask_gt > 0.5) * dist_mask).float()

                weight = torch.ones(geo_feat_i.size(0), device=self.device, dtype=torch.float32)  # b
                weight = weight.unsqueeze(-1).unsqueeze(-1)  # b,1,1

                # compute smooth l1 loss between nn pair
                raw_loss = F.smooth_l1_loss(geo_feat_i, nn_feat, reduction="none") * weight  # b,n,c
                loss += (raw_loss.mean(2) * pair_mask).sum() / (pair_mask.sum() + 1e-6)

        loss = loss / cnt if cnt > 0 else loss  # avoid division by zero

        return loss

    def build_memory_cycle_loss(self, input):
        fg_memory_tokens = input["fg_memory_tokens"]  # b,t-1,num_fg_tokens,c
        geo_feats = input["geo_feats"]  # b,t,c,n
        geo_feats = geo_feats.permute(0, 1, 3, 2)  # b,t,n,c
        mask_gts = input["mask_gts"]  # b,t,n

        B = fg_memory_tokens.size(0)  # batch size
        N_FG = fg_memory_tokens.size(2)  # number of foreground tokens

        temperature = self.cfg.loss_cfg.memory_cycle_temperature

        loss_a = 0.0
        loss_b = 0.0

        for ti in range(self.cfg.dataset_cfg.num_smp_frames_per_tracklet - 1):  # exclude the last frame
            mem = fg_memory_tokens[:, ti, :, :]  # b,num_tokens,c
            msk = mask_gts[:, ti, :]  # b,n
            has_fg = msk.sum(dim=1) > 0  # b
            has_bg = (1 - msk).sum(dim=1) > 0  # b

            # memory to point cosine similarity and affinity
            mem_pt_cos_sim = F.cosine_similarity(mem.unsqueeze(2), geo_feats[:, ti, :, :].unsqueeze(1), dim=-1)  # b,num_fg_tokens,n
            mem_pt_affinity = torch.softmax(mem_pt_cos_sim / temperature, dim=-1)  # b,num_fg_tokens,n

            # for foreground tokens, maximize the sum of affinity to foreground points
            fg_msk = msk.unsqueeze(1).expand(B, N_FG, -1)  # b,num_fg_tokens,n
            mem_pt_affinity_fg = mem_pt_affinity * fg_msk  # b,num_fg_tokens,n
            fg_loss = -torch.log(mem_pt_affinity_fg.sum(dim=2) + 1e-8)  # b,num_fg_tokens

            loss_a += (fg_loss * has_fg.float().unsqueeze(1) * has_bg.float().unsqueeze(1)).mean()  # b,num_fg_tokens

            # point to memory cosine similarity and affinity
            pt_mem_cos_sim = mem_pt_cos_sim.permute(0, 2, 1)  # b,n,num_fg_tokens
            pt_mem_affinity = torch.softmax(pt_mem_cos_sim / temperature, dim=-1)  # b,n,num_fg_tokens

            # memory cycle probability
            cycle_prob = torch.bmm(mem_pt_affinity, pt_mem_affinity)  # b,num_fg_tokens,num_fg_tokens

            # loss calculation
            target = torch.eye(mem.size(1), device=self.device, dtype=torch.float32)  # num_fg_tokens,num_fg_tokens
            target = target.unsqueeze(0).expand(B, -1, -1)  # b,num_tokens,num_tokens
            ce_loss = -torch.log(cycle_prob + 1e-8) * target  # b,num_tokens,num_tokens

            # if there is no foreground point in the batch, set loss of foreground tokens to 0 and vice versa
            loss_fg_mask = has_fg.unsqueeze(1).unsqueeze(2)  # b,1,1
            ce_loss = ce_loss * loss_fg_mask  # b,num_fg_tokens,num_fg_tokens
            loss_b += ce_loss.mean()

        loss = loss_a * self.cfg.loss_cfg.memory_cycle_fg_weight + loss_b * self.cfg.loss_cfg.memory_cycle_prob_weight
        loss = loss / (self.cfg.dataset_cfg.num_smp_frames_per_tracklet - 1)  # average over frames
        return loss

    def training_step(self, batch, batch_idx):
        pcds = batch["pcds"]  # b,t,N,3
        mask_gts = batch["mask_gts"]  # b,t,N
        bbox_gts = batch["bbox_gts"]  # b,t,4
        first_mask_gt = batch["first_mask_gt"]  # b,N
        lwh = batch["lwh"]  # b,3

        fg_memory_tokens_list = []

        embed_output = self.model(dict(pcds=pcds), mode="embed")
        xyzs = embed_output["xyzs"]  # b,t,n,3
        geo_feats = embed_output["feats"]  # b,t,c,n
        idxs = embed_output["idxs"]  # b,t,n

        update_output = self.model(
            dict(
                geo_feats=geo_feats[:, 0, :, :],  # b,c,n
                xyz=xyzs[:, 0, :, :],
                mask=torch.gather(first_mask_gt, 1, idxs[:, 0, :]),
            ),
            mode="update",
        )
        memory = update_output["memory"]
        fg_memory_tokens_list.append(memory["fg_tokens"])  # b,num_fg_tokens,c

        n_smp_frame = self.cfg.dataset_cfg.num_smp_frames_per_tracklet

        mask_loss, crs_obj_loss, rfn_obj_loss, center_loss, bbox_loss = 0.0, 0.0, 0.0, 0.0, 0.0
        temporal_consistency_loss = 0.0
        memory_cycle_loss = 0.0

        for i in range(1, n_smp_frame):
            propagte_output = self.model(dict(feat=geo_feats[:, i, :, :], xyz=xyzs[:, i, :, :], memory=memory), mode="propagate")
            prop_geo_feat = propagte_output["feat"]  # b, c, n

            localize_output = self.model(
                dict(geo_feat=prop_geo_feat, xyz=xyzs[:, i, :, :], lwh=lwh, center_gt=bbox_gts[:, i, :3], memory=memory),
                mode="localize",
            )
            mask_pred = localize_output["mask_pred"]  # b,n
            mask_loss += self.build_mask_loss(dict(pred=mask_pred, gt=torch.gather(mask_gts[:, i, :], 1, idxs[:, i, :])))
            center_pred = localize_output["center_pred"]
            center_loss += self.build_center_loss(
                dict(
                    pred=center_pred,
                    gt=bbox_gts[:, i, :3].unsqueeze(1).expand_as(center_pred),
                    mask=torch.gather(mask_gts[:, i, :], 1, idxs[:, i, :]),
                )
            )

            dist = torch.sum((center_pred - bbox_gts[:, i, None, :3]) ** 2, dim=-1)
            dist = torch.sqrt(dist + 1e-6)  # B, K
            objectness_label = torch.zeros_like(dist, dtype=torch.float)
            objectness_label[dist < 0.3] = 1
            objectness_mask = torch.ones_like(objectness_label, dtype=torch.float)
            objectness_pred = localize_output["objectness_pred"]
            crs_obj_loss += self.build_objectness_loss(dict(pred=objectness_pred, gt=objectness_label, mask=objectness_mask))

            bboxes_pred = localize_output["bboxes_pred"]
            proposal_xyz = localize_output["proposal_xyz"]
            dist = torch.sum((proposal_xyz - bbox_gts[:, i, None, :3]) ** 2, dim=-1)

            dist = torch.sqrt(dist + 1e-6)  # B, K
            objectness_label = torch.zeros_like(dist, dtype=torch.float)
            objectness_label[dist < 0.3] = 1
            objectness_pred = bboxes_pred[:, :, 4]  # B, K
            objectness_mask = torch.ones_like(objectness_label, dtype=torch.float)
            rfn_obj_loss += self.build_objectness_loss(dict(pred=objectness_pred, gt=objectness_label, mask=objectness_mask))
            bbox_loss += self.build_bbox_loss(
                dict(
                    pred=bboxes_pred[:, :, :4],
                    gt=bbox_gts[:, i, None, :4].expand_as(bboxes_pred[:, :, :4]),
                    mask=objectness_label,
                )
            )

            if i < n_smp_frame - 1:
                update_output = self.model(
                    dict(
                        geo_feats=geo_feats[:, i, :, :],
                        xyz=xyzs[:, i, :, :],
                        mask=mask_pred.sigmoid(),
                        memory=memory,
                    ),
                    mode="update",
                )
                memory = update_output["memory"]
                fg_memory_tokens_list.append(memory["fg_tokens"])  # b,num_fg_tokens,c

        if self.cfg.loss_cfg.temporal_consistency_weight > 0:
            temporal_consistency_loss += self.build_temporal_consistency_loss(
                dict(
                    geo_feats=geo_feats,
                    mask_gts=torch.gather(mask_gts, 2, idxs),
                    bbox_gts=bbox_gts,
                    xyzs=xyzs,
                )
            )

        if self.cfg.loss_cfg.memory_cycle_weight > 0.0:
            fg_memory_tokens = torch.stack(fg_memory_tokens_list, dim=1)  # b,t-1,num_fg_tokens,c

            memory_cycle_loss += self.build_memory_cycle_loss(
                dict(
                    fg_memory_tokens=fg_memory_tokens,
                    geo_feats=geo_feats,
                    mask_gts=torch.gather(mask_gts, 2, idxs),
                )
            )


        loss = (
            self.cfg.loss_cfg.mask_weight * mask_loss
            + self.cfg.loss_cfg.crs_obj_weight * crs_obj_loss
            + self.cfg.loss_cfg.rfn_obj_weight * rfn_obj_loss
            + self.cfg.loss_cfg.bbox_weight * bbox_loss
            + self.cfg.loss_cfg.center_weight * center_loss
            + self.cfg.loss_cfg.temporal_consistency_weight * temporal_consistency_loss
            + self.cfg.loss_cfg.memory_cycle_weight * memory_cycle_loss
        )

        if loss.isnan():
            print("loss is nan")

        # for detecting unused parameters
        # loss.backward()
        # for name, param in self.model.named_parameters():
        #     if param.grad is None:
        #         print(name)

        self.logger.experiment.add_scalars(
            "loss",
            {
                "loss_total": loss,
                "loss_bbox": bbox_loss,
                "loss_center": center_loss,
                "loss_mask": mask_loss,
                "loss_rfn_objectness": rfn_obj_loss,
                "loss_crs_objectness": crs_obj_loss,
                "loss_temporal_consistency": temporal_consistency_loss,
                "loss_memory_cycle": memory_cycle_loss,
            },
            global_step=self.global_step,
        )

        return loss

    def _to_float_tensor(self, data):
        tensor_data = {}
        for k, v in data.items():
            tensor_data[k] = torch.tensor(v, device=self.device, dtype=torch.float32).unsqueeze(0)
        return tensor_data

    def forward_on_tracklet(self, tracklet):
        pred_bboxes = []
        gt_bboxes = []

        memory = None
        lwh = None

        last_bbox_cpu = np.array([0.0, 0.0, 0.0, 0.0])

        for frame_id, frame in enumerate(tracklet):
            gt_bboxes.append(frame["bbox"])

            if frame_id == 0:
                base_bbox = frame["bbox"]
                lwh = np.array([base_bbox.wlh[1], base_bbox.wlh[0], base_bbox.wlh[2]])
            else:
                base_bbox = pred_bboxes[-1]

            # preprocess point cloud
            pcd = crop_and_center_pcd(
                frame["pcd"],
                base_bbox,
                offset=self.cfg.dataset_cfg.frame_offset,
                offset2=self.cfg.dataset_cfg.frame_offset2,
                scale=self.cfg.dataset_cfg.frame_scale,
            )

            if frame_id == 0:
                if pcd.nbr_points() == 0:
                    pcd.points = np.array([[0.0], [0.0], [0.0]])
                bbox = transform_box(frame["bbox"], base_bbox)  # get relative bbox
                mask_gt = get_pcd_in_box_mask(pcd, bbox, scale=self.cfg.dataset_cfg.mask_scale).astype(int)
                pcd, idx = resample_pcd(pcd, self.cfg.dataset_cfg.frame_npts, return_idx=True, is_training=False)
                mask_gt = mask_gt[idx]
            else:
                if pcd.nbr_points() <= 1:
                    bbox = get_offset_box(pred_bboxes[-1], last_bbox_cpu, use_z=self.cfg.dataset_cfg.eval_cfg.use_z, is_training=False)
                    pred_bboxes.append(bbox)
                    continue

                pcd, idx = resample_pcd(pcd, self.cfg.dataset_cfg.frame_npts, return_idx=True, is_training=False)

            embed_output = self.model(
                dict(pcds=torch.tensor(pcd.points.T, device=self.device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)),
                mode="embed",
            )
            xyzs = embed_output["xyzs"]  # (1, 1, n, 3)
            geo_feats = embed_output["feats"]  # (1, 1, c, n)
            idxs = embed_output["idxs"]  # (1, 1, n)

            if frame_id == 0:
                first_mask_gt = torch.tensor(mask_gt, device=self.device, dtype=torch.float32).unsqueeze(0)

                update_output = self.model(
                    dict(
                        geo_feats=geo_feats[:, 0, :, :],
                        xyz=xyzs[:, 0, :, :],
                        mask=torch.gather(first_mask_gt.squeeze(1), 1, idxs[:, 0, :]),
                    ),
                    mode="update",
                )
                memory = update_output["memory"]

                pred_bboxes.append(frame["bbox"])
            else:
                propagte_output = self.model(dict(feat=geo_feats[:, 0, :, :], xyz=xyzs[:, 0, :, :], memory=memory), mode="propagate")
                prop_geo_feat = propagte_output["feat"]  # b, c, n

                localize_output = self.model(
                    dict(
                        geo_feat=prop_geo_feat,  # b,c,n
                        xyz=xyzs[:, 0, :, :],
                        lwh=torch.tensor(lwh, device=self.device, dtype=torch.float32).unsqueeze(0),
                        memory=memory,
                    ),
                    mode="localize",
                )
                mask_pred = localize_output["mask_pred"]

                bboxes_pred = localize_output["bboxes_pred"]
                bboxes_pred_cpu = bboxes_pred.squeeze(0).detach().cpu().numpy()

                # remove bboxes whose objectness pred is nan
                # it may happen at the early stage of training
                bboxes_pred_cpu[np.isnan(bboxes_pred_cpu)] = -1e6

                best_box_idx = bboxes_pred_cpu[:, 4].argmax()
                bbox_cpu = bboxes_pred_cpu[best_box_idx, 0:4]

                if torch.max(mask_pred.sigmoid()) < self.cfg.missing_threshold:
                    bbox = get_offset_box(pred_bboxes[-1], last_bbox_cpu, use_z=self.cfg.dataset_cfg.eval_cfg.use_z, is_training=False)
                else:
                    bbox = get_offset_box(pred_bboxes[-1], bbox_cpu, use_z=self.cfg.dataset_cfg.eval_cfg.use_z, is_training=False)
                    last_bbox_cpu = bbox_cpu

                pred_bboxes.append(bbox)
                if frame_id < len(tracklet) - 1:
                    update_output = self.model(
                        dict(
                            geo_feats=geo_feats[:, 0, :, :],
                            xyz=xyzs[:, 0, :, :],
                            mask=mask_pred.sigmoid(),
                            memory=memory,
                        ),
                        mode="update",
                    )
                    memory = update_output["memory"]

        return pred_bboxes, gt_bboxes

