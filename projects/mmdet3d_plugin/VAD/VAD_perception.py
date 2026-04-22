import torch
import torch.nn as nn
import copy
from mmdet.models import DETECTORS, HEADS
from projects.mmdet3d_plugin.VAD.VAD import VAD
from projects.mmdet3d_plugin.VAD.VAD_head import VADHead
from mmdet3d.core import bbox3d2result
from mmcv.runner import force_fp32
from mmdet.core import multi_apply

@HEADS.register_module()
class VADPerceptionHead(VADHead):
    def __init__(self, *args, **kwargs):
        # Disable all motion and planning decoders
        kwargs['motion_decoder'] = None
        kwargs['motion_map_decoder'] = None
        kwargs['ego_his_encoder'] = None
        kwargs['ego_agent_decoder'] = None
        kwargs['ego_map_decoder'] = None
        super(VADPerceptionHead, self).__init__(*args, **kwargs)

    @force_fp32(apply_to=('mlvl_feats', 'prev_bev'))
    def forward(self,
                mlvl_feats,
                img_metas,
                prev_bev=None,
                only_bev=False,
                **kwargs):      # Catch other arguments like ego_his_trajs etc which we ignore.
        
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        object_query_embeds = self.query_embedding.weight.to(dtype)
        
        if self.map_query_embed_type == 'all_pts':
            map_query_embeds = self.map_query_embedding.weight.to(dtype)
        elif self.map_query_embed_type == 'instance_pts':
            map_pts_embeds = self.map_pts_embedding.weight.unsqueeze(0)
            map_instance_embeds = self.map_instance_embedding.weight.unsqueeze(1)
            map_query_embeds = (map_pts_embeds + map_instance_embeds).flatten(0, 1).to(dtype)

        bev_queries = self.bev_embedding.weight.to(dtype)
        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                               device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)
            
        if only_bev:
            return self.transformer.get_bev_features(
                mlvl_feats,
                bev_queries,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                img_metas=img_metas,
                prev_bev=prev_bev,
            )
        else:
            outputs = self.transformer(
                mlvl_feats,
                bev_queries,
                object_query_embeds,
                map_query_embeds,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                reg_branches=self.reg_branches if self.with_box_refine else None,
                cls_branches=self.cls_branches if self.as_two_stage else None,
                map_reg_branches=self.map_reg_branches if self.with_box_refine else None,
                map_cls_branches=self.map_cls_branches if self.as_two_stage else None,
                img_metas=img_metas,
                prev_bev=prev_bev
            )

        bev_embed, hs, init_reference, inter_references, \
            map_hs, map_init_reference, map_inter_references = outputs

        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []

        map_hs = map_hs.permute(0, 2, 1, 3)
        map_outputs_classes = []
        map_outputs_coords = []
        map_outputs_pts_coords = []

        from mmdet.models.utils.transformer import inverse_sigmoid
        
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])
            tmp[..., 0:2] = tmp[..., 0:2] + reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            tmp[..., 4:5] = tmp[..., 4:5] + reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
            tmp[..., 0:1] = (tmp[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0])
            tmp[..., 1:2] = (tmp[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1])
            tmp[..., 4:5] = (tmp[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2])
            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        
        for lvl in range(map_hs.shape[0]):
            if lvl == 0:
                reference = map_init_reference
            else:
                reference = map_inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            map_outputs_class = self.map_cls_branches[lvl](
                map_hs[lvl].view(bs, self.map_num_vec, self.map_num_pts_per_vec, -1).mean(2)
            )
            tmp = self.map_reg_branches[lvl](map_hs[lvl])
            tmp[..., 0:2] += reference[..., 0:2]
            tmp = tmp.sigmoid() # cx,cy,w,h
            map_outputs_coord, map_outputs_pts_coord = self.map_transform_box(tmp)
            map_outputs_classes.append(map_outputs_class)
            map_outputs_coords.append(map_outputs_coord)
            map_outputs_pts_coords.append(map_outputs_pts_coord)

        map_outputs_classes = torch.stack(map_outputs_classes)
        map_outputs_coords = torch.stack(map_outputs_coords)
        map_outputs_pts_coords = torch.stack(map_outputs_pts_coords)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)

        # Truncate here, do not execute the complex planner and motion prediction branches!
        # Provide dummy values for trajs to satisfy CustomNMSFreeCoder
        num_dec = outputs_coords.shape[0]
        num_query = outputs_coords.shape[2]
        dummy_trajs = torch.zeros(num_dec, bs, num_query, self.fut_ts*2, device=bev_embed.device)
        dummy_traj_cls = torch.zeros(num_dec, bs, num_query, self.fut_mode, device=bev_embed.device)

        outs = {
            'bev_embed': bev_embed,
            'all_cls_scores': outputs_classes,
            'all_bbox_preds': outputs_coords,
            'all_traj_preds': dummy_trajs, # dummy
            'all_traj_cls_scores': dummy_traj_cls, # dummy
            'map_all_cls_scores': map_outputs_classes,
            'map_all_bbox_preds': map_outputs_coords,
            'map_all_pts_preds': map_outputs_pts_coords,
        }
        return outs

    def loss_single_perception(self,
                               cls_scores,
                               bbox_preds,
                               gt_bboxes_list,
                               gt_labels_list,
                               gt_attr_labels_list,
                               gt_bboxes_ignore_list=None):
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list,
                                           gt_attr_labels_list, gt_bboxes_ignore_list)

        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         traj_targets_list, traj_weights_list, gt_fut_masks_list,
         num_total_pos, num_total_neg) = cls_reg_targets

        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        from mmdet.core import reduce_mean
        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights
        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :10],
            normalized_bbox_targets[isnotnan, :10],
            bbox_weights[isnotnan, :10],
            avg_factor=num_total_pos)

        return loss_cls, loss_bbox

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             map_gt_bboxes_list,
             map_gt_labels_list,
             preds_dicts,
             gt_attr_labels,
             gt_bboxes_ignore=None,
             map_gt_bboxes_ignore=None,
             img_metas=None,
             **kwargs): # Catch unused ego planning args
             
        map_gt_vecs_list = copy.deepcopy(map_gt_bboxes_list)

        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']
        map_all_cls_scores = preds_dicts['map_all_cls_scores']
        map_all_bbox_preds = preds_dicts['map_all_bbox_preds']
        map_all_pts_preds = preds_dicts['map_all_pts_preds']

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device
        gt_bboxes_list = [torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]), dim=1).to(device) for gt_bboxes in gt_bboxes_list]
        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_attr_labels_list = [gt_attr_labels for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [gt_bboxes_ignore for _ in range(num_dec_layers)]

        from mmdet.core import multi_apply
        losses_cls, losses_bbox = multi_apply(
            self.loss_single_perception, all_cls_scores, all_bbox_preds, 
            all_gt_bboxes_list, all_gt_labels_list, all_gt_attr_labels_list, all_gt_bboxes_ignore_list)
        
        num_dec_layers = len(map_all_cls_scores)
        device = map_gt_labels_list[0].device
        map_gt_bboxes_list_boxes = [map_gt_bboxes.bbox.to(device) for map_gt_bboxes in map_gt_vecs_list]
        
        if self.map_gt_shift_pts_pattern == 'v0':
            map_gt_shifts_pts_list = [gt_bboxes.shift_fixed_num_sampled_points.to(device) for gt_bboxes in map_gt_vecs_list]
        elif self.map_gt_shift_pts_pattern == 'v1':
            map_gt_shifts_pts_list = [gt_bboxes.shift_fixed_num_sampled_points_v1.to(device) for gt_bboxes in map_gt_vecs_list]
        elif self.map_gt_shift_pts_pattern == 'v2':
            map_gt_shifts_pts_list = [gt_bboxes.shift_fixed_num_sampled_points_v2.to(device) for gt_bboxes in map_gt_vecs_list]
        elif self.map_gt_shift_pts_pattern == 'v3':
            map_gt_shifts_pts_list = [gt_bboxes.shift_fixed_num_sampled_points_v3.to(device) for gt_bboxes in map_gt_vecs_list]
        elif self.map_gt_shift_pts_pattern == 'v4':
            map_gt_shifts_pts_list = [gt_bboxes.shift_fixed_num_sampled_points_v4.to(device) for gt_bboxes in map_gt_vecs_list]

        map_all_gt_bboxes_list = [map_gt_bboxes_list_boxes for _ in range(num_dec_layers)]
        map_all_gt_labels_list = [map_gt_labels_list for _ in range(num_dec_layers)]
        map_all_gt_shifts_pts_list = [map_gt_shifts_pts_list for _ in range(num_dec_layers)]
        map_all_gt_bboxes_ignore_list = [map_gt_bboxes_ignore for _ in range(num_dec_layers)]

        map_losses_cls, map_losses_bbox, map_losses_iou, map_losses_pts, map_losses_dir = multi_apply(
            self.map_loss_single, map_all_cls_scores, map_all_bbox_preds, map_all_pts_preds,
            map_all_gt_bboxes_list, map_all_gt_labels_list, map_all_gt_shifts_pts_list,
            map_all_gt_bboxes_ignore_list)

        loss_dict = dict()
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_map_cls'] = map_losses_cls[-1]
        loss_dict['loss_map_bbox'] = map_losses_bbox[-1]
        loss_dict['loss_map_iou'] = map_losses_iou[-1]
        loss_dict['loss_map_pts'] = map_losses_pts[-1]
        loss_dict['loss_map_dir'] = map_losses_dir[-1]

        for num_dec_layer, (loss_cls_i, loss_bbox_i) in enumerate(zip(losses_cls[:-1], losses_bbox[:-1])):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            
        for num_dec_layer, (map_loss_cls_i, map_loss_bbox_i, map_loss_iou_i, map_loss_pts_i, map_loss_dir_i) in \
            enumerate(zip(map_losses_cls[:-1], map_losses_bbox[:-1], map_losses_iou[:-1], map_losses_pts[:-1], map_losses_dir[:-1])):
            loss_dict[f'd{num_dec_layer}.loss_map_cls'] = map_loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_map_bbox'] = map_loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_map_iou'] = map_loss_iou_i
            loss_dict[f'd{num_dec_layer}.loss_map_pts'] = map_loss_pts_i
            loss_dict[f'd{num_dec_layer}.loss_map_dir'] = map_loss_dir_i

        return loss_dict

@DETECTORS.register_module()
class VADPerception(VAD):
    
    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          map_gt_bboxes_3d,
                          map_gt_labels_3d,                          
                          img_metas,
                          gt_bboxes_ignore=None,
                          map_gt_bboxes_ignore=None,
                          prev_bev=None,
                          gt_attr_labels=None,
                          **kwargs): # catch fut commands which we don't need
        outs = self.pts_bbox_head(pts_feats, img_metas, prev_bev)
        loss_inputs = [
            gt_bboxes_3d, gt_labels_3d, map_gt_bboxes_3d, map_gt_labels_3d,
            outs, gt_attr_labels
        ]
        losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)
        return losses

    def simple_test(
        self,
        img_metas,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        img=None,
        prev_bev=None,
        rescale=False,
        **kwargs
    ):
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        bbox_list = [dict() for i in range(len(img_metas))]
        new_prev_bev, bbox_pts = self.simple_test_pts(
            img_feats,
            img_metas,
            prev_bev,
            rescale=rescale,
        )
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
            result_dict['metric_results'] = dict() # We disabled metrics for perception

        return new_prev_bev, bbox_list

    def simple_test_pts(
        self,
        x,
        img_metas,
        prev_bev=None,
        rescale=False,
    ):
        outs = self.pts_bbox_head(x, img_metas, prev_bev=prev_bev)
        bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas, rescale=rescale)

        bbox_results = []
        for i, (bboxes, scores, labels, trajs, map_bboxes, \
                map_scores, map_labels, map_pts) in enumerate(bbox_list):
            bbox_result = bbox3d2result(bboxes, scores, labels)
            map_bbox_result = self.map_pred2result(map_bboxes, map_scores, map_labels, map_pts)
            bbox_result.update(map_bbox_result)
            bbox_results.append(bbox_result)

        assert len(bbox_results) == 1, 'only support batch_size=1 now'
        score_threshold = 0.6
        with torch.no_grad():
            c_bbox_results = copy.deepcopy(bbox_results)
            bbox_result = c_bbox_results[0]
            mask = bbox_result['scores_3d'] > score_threshold
            bbox_result['boxes_3d'] = bbox_result['boxes_3d'][mask]
            bbox_result['scores_3d'] = bbox_result['scores_3d'][mask]
            bbox_result['labels_3d'] = bbox_result['labels_3d'][mask]
            # No EPA/Planning metrics computed.

        return outs['bev_embed'], bbox_results

