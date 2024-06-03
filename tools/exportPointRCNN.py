
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
from pcdet.ops import pointnet2
from pcdet.ops.pointnet2.pointnet2_batch import pointnet2_modules
from pcdet.ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_modules_stack
from pcdet.ops.pointnet2.pointnet2_stack import pointnet2_utils as pointnet2_utils_stack


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

        # for i in range(-1, -(len(self.FP_modules) + 1), -1):
        #     l_features[i - 1] = self.FP_modules[i](
        #         l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
        #     )  # (B, C, N)

        point_features = l_features[1]  # (B, C, N)
        return point_features


class PointHeadBox(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg

    def forward(self, feature):
        """
        extract class feature by cls_layers and box feature by box_layers
        :param feature: the extracted feature from PointNet2MSG, (B, C, N)
        :return: cls_feature (B, N_cls, N), box_feature (B, 8, N), 8 is center(x,y,z) , w, h, l, angle and score
        """
        return


class PointRCNNHead(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg

    def forward(self, xyz, point_feature, cls_feature, box_feature):
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
        return


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
    new_model = NewModel(model)
    print(new_model)
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
    batch_idx, xyz, features = break_up_pc(points)
    # print("batch_size: ", batch_size)
    xyz_batch_cnt = xyz.new_zeros(batch_size).int()
    for bs_idx in range(batch_size):
        xyz_batch_cnt[bs_idx] = (batch_idx == bs_idx).sum()
    # print("min: ", xyz_batch_cnt.min(), " max: ", xyz_batch_cnt.max())
    assert xyz_batch_cnt.min() == xyz_batch_cnt.max()
    xyz = xyz.view(batch_size, -1, 3)
    features = features.view(batch_size, -1, features.shape[-1]).permute(0, 2, 1).contiguous() if features is not None else None
    dicts = {}
    checkpoint = torch.load(args.ckpt, map_location='cuda')
    for key in checkpoint['model_state'].keys():
        # print(key)
        if "backbone_3d" in key:
            dicts[key[12:]] = checkpoint['model_state'][key]

    pointNet2_export = PointNet2MSGExport(cfg.MODEL.BACKBONE_3D, 4)
    pointNet2_export.load_state_dict(dicts)
    pointNet2_export.cuda()
    pointNet2_export.eval()
    out = pointNet2_export(xyz, features)
    print(out)
    onnx_path = "./test.onnx"
    print("start convert model to onnx >>>")
    # torch.onnx.export(new_model,  # support torch.nn.Module, torch.jit.ScriptModule or torch.jit.ScriptFunction
    #                   (inp,),
    #                   onnx_path,
    #                   verbose=True,
    #                   input_names=["points"],
    #                   output_names=["select_ids"],
    #                   opset_version=12,
    #                   operator_export_type=torch.onnx.OperatorExportTypes.ONNX,  # ONNX_ATEN_FALLBACK,
    #                   dynamic_axes={
    #                       "points": {1: "b", 2: "c", 3: "n"},
    #                       "select_ids": {0: "b", 1: "n", 2: "c"}
    #                   }
    # )
    torch.onnx.export(pointNet2_export,  # support torch.nn.Module, torch.jit.ScriptModule or torch.jit.ScriptFunction
                      (xyz, features, ),
                      onnx_path,
                      verbose=True,
                      input_names=["points", "features"],
                      output_names=["pointnet2_features"],
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
