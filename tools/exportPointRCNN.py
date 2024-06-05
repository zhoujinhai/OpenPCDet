
import argparse
import datetime
import glob
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from pcdet.ops.pointnet2.pointnet2_batch import pointnet2_modules
from pcdet.models.dense_heads.point_head_template import PointHeadTemplate
from pcdet.utils import box_coder_utils
from pcdet.models.roi_heads.pointrcnn_head import RoIHeadTemplate
from pcdet.ops.roipoint_pool3d import roipoint_pool3d_utils
from pcdet.models.model_utils.model_nms_utils import class_agnostic_nms


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=30, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--eval_tag', type=str, default='default', help='eval tag for this experiment')
    parser.add_argument('--eval_all', action='store_true', default=False, help='whether to evaluate all checkpoints')
    parser.add_argument('--ckpt_dir', type=str, default=None, help='specify a ckpt directory to be evaluated if needed')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    parser.add_argument('--infer_time', action='store_true', default=False, help='calculate inference latency')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    np.random.seed(1024)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


class PointNet2MSGExport(nn.Module):
    def __init__(self, model_cfg, input_channels, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg

        self.SA_modules = nn.ModuleList()
        channel_in = input_channels - 3

        self.num_points_each_layer = []
        skip_channel_list = [input_channels - 3]

        for k in range(self.model_cfg.SA_CONFIG.NPOINTS.__len__()):
            mlps = self.model_cfg.SA_CONFIG.MLPS[k].copy()
            channel_out = 0
            for idx in range(mlps.__len__()):
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]

            self.SA_modules.append(
                pointnet2_modules.PointnetSAModuleMSG(
                    npoint=self.model_cfg.SA_CONFIG.NPOINTS[k],
                    radii=self.model_cfg.SA_CONFIG.RADIUS[k],
                    nsamples=self.model_cfg.SA_CONFIG.NSAMPLE[k],
                    mlps=mlps,
                    use_xyz=self.model_cfg.SA_CONFIG.get('USE_XYZ', True),
                )
            )
            skip_channel_list.append(channel_out)
            channel_in = channel_out

        self.FP_modules = nn.ModuleList()

        for k in range(self.model_cfg.FP_MLPS.__len__()):
            pre_channel = self.model_cfg.FP_MLPS[k + 1][-1] if k + 1 < len(self.model_cfg.FP_MLPS) else channel_out
            self.FP_modules.append(
                pointnet2_modules.PointnetFPModule(
                    mlp=[pre_channel + skip_channel_list[k]] + self.model_cfg.FP_MLPS[k]
                )
            )

        self.num_point_features = self.model_cfg.FP_MLPS[0][-1]

    def forward(self, xyz, features):
        """
        xyz, feature ==> SA0 ==> SA1 ==> ... ==> SA_n ==> FP_n ==> ... ==> FP1 ==> FP0
        :param xyz: the coords of point, (B, N, 3)
        :param features: the feature of point, (B, C, N)
        :return: extracted feature, (B, C_out, N)
        """
        l_xyz = [xyz]
        l_features = [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )  # (B, C, N)

        l_features[0] = l_features[0].permute(0, 2, 1).contiguous()  # (B, C, N)
        l_features[0] = l_features[0].view(-1, l_features[0].shape[-1])
        return l_features[0]


class PointHeadBoxExport(PointHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class):
        super().__init__(model_cfg=model_cfg, num_class=num_class)
        self.model_cfg = model_cfg
        self.cls_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.CLS_FC,
            input_channels=input_channels,
            output_channels=num_class
        )

        target_cfg = self.model_cfg.TARGET_CONFIG
        self.box_coder = getattr(box_coder_utils, target_cfg.BOX_CODER)(
            **target_cfg.BOX_CODER_CONFIG
        )
        self.box_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.REG_FC,
            input_channels=input_channels,
            output_channels=self.box_coder.code_size
        )

    def forward(self, xyz, feature):
        """
        extract class feature by cls_layers and box feature by box_layers
        :param xyz: the coords of points
        :param feature: the extracted feature from PointNet2MSG, (B, C, N)
        :return: cls_feature (B, N_cls, N), box_feature (B, 8, N), 8 is center(x,y,z) , w, h, l, angle and score
        """
        point_cls_preds = self.cls_layers(feature)  # (total_points, num_class)
        point_box_preds = self.box_layers(feature)  # (total_points, box_code_size)

        point_cls_preds_max, _ = point_cls_preds.max(dim=-1)
        point_cls_scores = torch.sigmoid(point_cls_preds_max)

        point_cls_preds, point_box_preds = self.generate_predicted_boxes(
            points=xyz,
            point_cls_preds=point_cls_preds, point_box_preds=point_box_preds
        )

        return point_cls_scores, point_cls_preds, point_box_preds


class PointRCNNHeadExport(RoIHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, batch_size=2):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg
        self.batch_size = batch_size
        use_bn = self.model_cfg.USE_BN
        self.SA_modules = nn.ModuleList()
        channel_in = input_channels

        self.num_prefix_channels = 3 + 2  # xyz + point_scores + point_depth
        xyz_mlps = [self.num_prefix_channels] + self.model_cfg.XYZ_UP_LAYER
        shared_mlps = []
        for k in range(len(xyz_mlps) - 1):
            shared_mlps.append(nn.Conv2d(xyz_mlps[k], xyz_mlps[k + 1], kernel_size=1, bias=not use_bn))
            if use_bn:
                shared_mlps.append(nn.BatchNorm2d(xyz_mlps[k + 1]))
            shared_mlps.append(nn.ReLU())
        self.xyz_up_layer = nn.Sequential(*shared_mlps)

        c_out = self.model_cfg.XYZ_UP_LAYER[-1]
        self.merge_down_layer = nn.Sequential(
            nn.Conv2d(c_out * 2, c_out, kernel_size=1, bias=not use_bn),
            *[nn.BatchNorm2d(c_out), nn.ReLU()] if use_bn else [nn.ReLU()]
        )

        for k in range(self.model_cfg.SA_CONFIG.NPOINTS.__len__()):
            mlps = [channel_in] + self.model_cfg.SA_CONFIG.MLPS[k]

            npoint = self.model_cfg.SA_CONFIG.NPOINTS[k] if self.model_cfg.SA_CONFIG.NPOINTS[k] != -1 else None
            self.SA_modules.append(
                pointnet2_modules.PointnetSAModule(
                    npoint=npoint,
                    radius=self.model_cfg.SA_CONFIG.RADIUS[k],
                    nsample=self.model_cfg.SA_CONFIG.NSAMPLE[k],
                    mlp=mlps,
                    use_xyz=True,
                    bn=use_bn
                )
            )
            channel_in = mlps[-1]

        self.cls_layers = self.make_fc_layers(
            input_channels=channel_in, output_channels=self.num_class, fc_list=self.model_cfg.CLS_FC
        )
        self.reg_layers = self.make_fc_layers(
            input_channels=channel_in,
            output_channels=self.box_coder.code_size * self.num_class,
            fc_list=self.model_cfg.REG_FC
        )

        self.roipoint_pool3d_layer = roipoint_pool3d_utils.RoIPointPool3d(
            num_sampled_points=self.model_cfg.ROI_POINT_POOL.NUM_SAMPLED_POINTS,
            pool_extra_width=self.model_cfg.ROI_POINT_POOL.POOL_EXTRA_WIDTH
        )

    @torch.no_grad()
    def proposal_layer_export(self, batch_idx, batch_box_preds, batch_cls_preds, nms_config):
        """
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
            nms_config:

        Returns:
            batch_dict:
                rois: (B, num_rois, 7+C)
                roi_scores: (B, num_rois)
                roi_labels: (B, num_rois)

        """
        rois = batch_box_preds.new_zeros((self.batch_size, nms_config.NMS_POST_MAXSIZE, batch_box_preds.shape[-1]))
        roi_scores = batch_box_preds.new_zeros((self.batch_size, nms_config.NMS_POST_MAXSIZE))
        roi_labels = batch_box_preds.new_zeros((self.batch_size, nms_config.NMS_POST_MAXSIZE), dtype=torch.long)

        for index in range(self.batch_size):
            batch_mask = (batch_idx == index)

            box_preds = batch_box_preds[batch_mask]
            cls_preds = batch_cls_preds[batch_mask]

            cur_roi_scores, cur_roi_labels = torch.max(cls_preds, dim=1)

            selected, selected_scores = class_agnostic_nms(
                    box_scores=cur_roi_scores, box_preds=box_preds, nms_config=nms_config
                )

            rois[index, :len(selected), :] = box_preds[selected]
            roi_scores[index, :len(selected)] = cur_roi_scores[selected]
            roi_labels[index, :len(selected)] = cur_roi_labels[selected]

        return rois, roi_scores, roi_labels + 1

    def roipool3d_gpu(self, batch_idx, point_coords, point_features, rois, point_scores):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                point_coords: (num_points, 4)  [bs_idx, x, y, z]
                point_features: (num_points, C)
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        Returns:

        """
        batch_cnt = point_coords.new_zeros(self.batch_size).int()
        for bs_idx in range(self.batch_size):
            batch_cnt[bs_idx] = (batch_idx == bs_idx).sum()

        assert batch_cnt.min() == batch_cnt.max()

        # point_scores = batch_dict['point_cls_scores'].detach()
        point_depths = point_coords.norm(dim=1) / self.model_cfg.ROI_POINT_POOL.DEPTH_NORMALIZER - 0.5
        point_features_list = [point_scores[:, None], point_depths[:, None], point_features]
        point_features_all = torch.cat(point_features_list, dim=1)
        batch_points = point_coords.view(self.batch_size, -1, 3)
        batch_point_features = point_features_all.view(self.batch_size, -1, point_features_all.shape[-1])

        with torch.no_grad():
            pooled_features, pooled_empty_flag = self.roipoint_pool3d_layer(
                batch_points, batch_point_features, rois
            )  # pooled_features: (B, num_rois, num_sampled_points, 3 + C), pooled_empty_flag: (B, num_rois)

            # canonical transformation
            roi_center = rois[:, :, 0:3]
            pooled_features[:, :, :, 0:3] -= roi_center.unsqueeze(dim=2)

            pooled_features = pooled_features.view(-1, pooled_features.shape[-2], pooled_features.shape[-1])
            pooled_features[:, :, 0:3] = common_utils.rotate_points_along_z(
                pooled_features[:, :, 0:3], -rois.view(-1, rois.shape[-1])[:, 6]
            )
            pooled_features[pooled_empty_flag.view(-1) > 0] = 0
        return pooled_features

    def forward(self, batch_idx, xyz, point_features, point_cls_scores, cls_feature, box_feature):
        """
        generate roi(B, N_roi, 7), score(B, N_roi, 1), roi_labels(B, N_roi, 1) by proposal_layer
        assign feature to every roi by roipool3d_layer  pooled_feature(B, N_roi, C, N_down)
        xyz_up_layer, input is the front five channel of pooled_feature, output is up_feature(B, N_roi, C_out, N_down)
        merge_down_layer, input is concat the remain channel of pooled_feature and the up_feature,
                          output is f(B, N_roi, C_out, N_down)
        SA_layer: Input is xyz and f ==> output f_out(B, N_roi, N_down, 1)
        cls_layer: output the class information
        reg_layer: output the box information
        :param xyz:  the coords of point (B, N, 3)
        :param point_feature: the feature of point extract by pointNet2MSG module, (B, N, C)
        :param cls_feature: the class feature (B, N, 1)
        :param box_feature: the box feature(B, N, 8), box info + score
        :return: cls_feature(B, N_roi, N_class), reg_feature(B, N_roi, 7)
        """
        rois, roi_scores, roi_labels = self.proposal_layer_export(
            batch_idx, box_feature, cls_feature, nms_config=self.model_cfg.NMS_CONFIG['TEST']
        )
        pooled_features = self.roipool3d_gpu(batch_idx, xyz, point_features, rois, point_cls_scores)
        xyz_input = pooled_features[..., 0:self.num_prefix_channels].transpose(1, 2).unsqueeze(dim=3).contiguous()
        xyz_features = self.xyz_up_layer(xyz_input)
        point_features = pooled_features[..., self.num_prefix_channels:].transpose(1, 2).unsqueeze(dim=3)
        merged_features = torch.cat((xyz_features, point_features), dim=1)
        merged_features = self.merge_down_layer(merged_features)

        l_xyz, l_features = [pooled_features[..., 0:3].contiguous()], [merged_features.squeeze(dim=3).contiguous()]

        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        shared_features = l_features[-1]  # (total_rois, num_features, 1)
        rcnn_cls = self.cls_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, 1 or 2)
        rcnn_reg = self.reg_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, C)

        batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
            batch_size=self.batch_size, rois=rois, cls_preds=rcnn_cls, box_preds=rcnn_reg
        )

        return batch_cls_preds, batch_box_preds


class NewModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.sa = model.backbone_3d.SA_modules[0]

    def forward(self, input):
        out = self.sa(input)
        return out


def break_up_pc(pc):
    batch_idx = pc[:, 0]
    xyz = pc[:, 1:4].contiguous()
    features = (pc[:, 4:].contiguous() if pc.size(-1) > 4 else None)
    return batch_idx, xyz, features


def export_single_ckpt(model, test_loader, args, eval_output_dir, logger, epoch_id, dist_test=False):
    # load checkpoint
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=dist_test,
                                pre_trained_path=args.pretrained_model)
    model.cuda()
    model.eval()

    # print("------------------")
    # print(dir(model))
    # print(model.module_list)
    # print(len(model.module_list))
    # print("******************")
    # # for idx, m in enumerate(model.named_modules()):
    # #     print(idx, "-", m)
    # new_model = NewModel(model)
    # print(new_model)
    print("-----------------")
    inp = None
    for i, batch_dict in enumerate(test_loader):
        load_data_to_gpu(batch_dict)
        inp = batch_dict
        print("$$$$$$$$$$$$", type(batch_dict))  # $$$$$$$$$$$$ <class 'dict'>
        break
    # out = model(inp)
    # print(out)
    batch_size = inp['batch_size']
    points = inp['points']
    # print("points size: ", points.shape, len(points[points[:, 0] == 0]), len(points[points[:, 0] == 1]))
    batch_idx, xyz_, features = break_up_pc(points)
    # print("batch_size: ", batch_size)
    xyz_batch_cnt = xyz_.new_zeros(batch_size).int()
    for bs_idx in range(batch_size):
        xyz_batch_cnt[bs_idx] = (batch_idx == bs_idx).sum()
    # print("min: ", xyz_batch_cnt.min(), " max: ", xyz_batch_cnt.max())
    assert xyz_batch_cnt.min() == xyz_batch_cnt.max()
    xyz = xyz_.view(batch_size, -1, 3)
    features = features.view(batch_size, -1, features.shape[-1]).permute(0, 2, 1).contiguous() if features is not None else None
    dicts = {}
    checkpoint = torch.load(args.ckpt, map_location='cuda')
    for key in checkpoint['model_state'].keys():
        print(key)
        if "backbone_3d" in key:
            dicts[key[12:]] = checkpoint['model_state'][key]

    pointNet2_export = PointNet2MSGExport(cfg.MODEL.BACKBONE_3D, 4)
    pointNet2_export.load_state_dict(dicts)
    pointNet2_export.cuda()
    pointNet2_export.eval()
    out1 = pointNet2_export(xyz, features)
    print("out1: ", out1)

    point_head_box_export = PointHeadBoxExport(cfg.MODEL.POINT_HEAD, out1.shape[-1], len(cfg.CLASS_NAMES))
    dicts = {}
    for key in checkpoint['model_state'].keys():
        # print(key)
        if "point_head" in key:
            dicts[key[11:]] = checkpoint['model_state'][key]
    point_head_box_export.load_state_dict(dicts)
    point_head_box_export.cuda()
    point_head_box_export.eval()
    point_cls_scores, point_cls_preds, point_box_preds = point_head_box_export(xyz_, out1)
    print(point_cls_scores, point_cls_preds, point_box_preds)

    point_rcnn_head_export = PointRCNNHeadExport(cfg.MODEL.ROI_HEAD, out1.shape[-1], len(cfg.CLASS_NAMES), batch_size)
    dicts = {}
    for key in checkpoint['model_state'].keys():
        # print(key)
        if "roi_head" in key:
            dicts[key[9:]] = checkpoint['model_state'][key]
    point_rcnn_head_export.load_state_dict(dicts)
    point_rcnn_head_export.cuda()
    point_rcnn_head_export.eval()
    out3 = point_rcnn_head_export(batch_idx, xyz_, out1, point_cls_scores, point_cls_preds, point_box_preds)
    print("out3: ", out3)
    onnx_msg_path = "./msg.onnx"
    print("start convert msg model to onnx >>>")

    torch.onnx.export(pointNet2_export,  # support torch.nn.Module, torch.jit.ScriptModule or torch.jit.ScriptFunction
                      (xyz, features, ),
                      onnx_msg_path,
                      verbose=True,
                      input_names=["points", "features"],
                      output_names=["pointnet2_features"],
                      opset_version=12,
                      operator_export_type=torch.onnx.OperatorExportTypes.ONNX,  # ONNX_ATEN_FALLBACK,
                      enable_onnx_checker=False
                      )

    print("start convert point_head_box model to onnx >>>")
    onnx_point_head_box_path = "./head_box.onnx"
    torch.onnx.export(point_head_box_export,  # support torch.nn.Module, torch.jit.ScriptModule or torch.jit.ScriptFunction
                      (xyz_, out1,),
                      onnx_point_head_box_path,
                      verbose=True,
                      input_names=["points", "features"],
                      output_names=["point_cls_scores", "point_cls_preds", "point_box_preds"],
                      opset_version=12,
                      operator_export_type=torch.onnx.OperatorExportTypes.ONNX,  # ONNX_ATEN_FALLBACK,
                      enable_onnx_checker=False
                      )

    print("start convert point_rcnn_head model to onnx >>>")
    onnx_point_rcnn_head_box_path = "./point_rcnn_head.onnx"
    torch.onnx.export(point_rcnn_head_export,
                      (batch_idx, xyz_, out1, point_cls_scores, point_cls_preds, point_box_preds,),
                      onnx_point_rcnn_head_box_path,
                      verbose=True,
                      input_names=["batch_idx", "points", "features", "pt_cls_scores", "pt_cls", "pt_box"],
                      output_names=["batch_cls_preds", "batch_box_preds"],
                      opset_version=12,
                      operator_export_type=torch.onnx.OperatorExportTypes.ONNX,  # ONNX_ATEN_FALLBACK,
                      enable_onnx_checker=False
                      )
    print("onnx model has exported!")


def get_no_evaluated_ckpt(ckpt_dir, ckpt_record_file, args):
    ckpt_list = glob.glob(os.path.join(ckpt_dir, '*checkpoint_epoch_*.pth'))
    ckpt_list.sort(key=os.path.getmtime)
    evaluated_ckpt_list = [float(x.strip()) for x in open(ckpt_record_file, 'r').readlines()]

    for cur_ckpt in ckpt_list:
        num_list = re.findall('checkpoint_epoch_(.*).pth', cur_ckpt)
        if num_list.__len__() == 0:
            continue

        epoch_id = num_list[-1]
        if 'optim' in epoch_id:
            continue
        if float(epoch_id) not in evaluated_ckpt_list and int(float(epoch_id)) >= args.start_epoch:
            return epoch_id, cur_ckpt
    return -1, None


def main():
    args, cfg = parse_config()

    if args.infer_time:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    if args.launcher == 'none':
        dist_test = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_test = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_output_dir = output_dir / 'eval'

    if not args.eval_all:
        num_list = re.findall(r'\d+', args.ckpt) if args.ckpt is not None else []
        epoch_id = num_list[-1] if num_list.__len__() > 0 else 'no_number'
        eval_output_dir = eval_output_dir / ('epoch_%s' % epoch_id) / cfg.DATA_CONFIG.DATA_SPLIT['test']
    else:
        eval_output_dir = eval_output_dir / 'eval_all_default'

    if args.eval_tag is not None:
        eval_output_dir = eval_output_dir / args.eval_tag

    eval_output_dir.mkdir(parents=True, exist_ok=True)
    log_file = eval_output_dir / ('log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_test:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)

    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_test, workers=args.workers, logger=logger, training=False
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)

    with torch.no_grad():
        export_single_ckpt(model, test_loader, args, eval_output_dir, logger, epoch_id, dist_test=dist_test)


if __name__ == '__main__':
    main()
