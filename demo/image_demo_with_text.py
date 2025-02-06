import os
import sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import gc
#import resource
import argparse
import cv2
import tqdm

import torch
from torch.multiprocessing import Pool, set_start_method

import mmcv
from mmcv.transforms import Compose
from mmengine.utils import track_iter_progress
from mmdet.apis import init_detector
from mmdet.registry import VISUALIZERS
from mmcv.ops.nms import batched_nms

import masa
from masa.apis import inference_masa, init_masa, inference_detector, build_test_pipeline
from masa.models.sam import SamPredictor, sam_model_registry
from utils import filter_and_update_tracks

import warnings
warnings.filterwarnings('ignore')

# Ensure the right start method for multiprocessing
try:
    set_start_method('spawn')
except RuntimeError:
    pass

""" def set_file_descriptor_limit(limit):
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (limit, hard))

# Set the file descriptor limit to 65536
set_file_descriptor_limit(65536) """

def visualize_frame(args, visualizer, frame, track_result, frame_idx, fps=None):
    visualizer.add_datasample(
        name='video_' + str(frame_idx),
        image=frame[:, :, ::-1],
        data_sample=track_result[0],
        draw_gt=False,
        show=False,
        out_file=None,
        pred_score_thr=args.score_thr,
        fps=fps,)
    frame = visualizer.get_image()
    gc.collect()
    return frame

def parse_args():

    parser = argparse.ArgumentParser(description='MASA video demo')
    parser.add_argument('video', help='Video file')
    parser.add_argument('--det_config', help='Detector Config file')
    parser.add_argument('--masa_config', help='Masa Config file')
    parser.add_argument('--det_checkpoint', help='Detector Checkpoint file')
    parser.add_argument('--masa_checkpoint', help='Masa Checkpoint file')
    parser.add_argument( '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--score-thr', type=float, default=0.2, help='Bbox score threshold')
    parser.add_argument('--out', type=str, help='Output video file')
    parser.add_argument('--save_dir', type=str, help='Output for video frames')
    parser.add_argument('--texts', help='text prompt')
    parser.add_argument('--line_width', type=int, default=5, help='Line width')
    parser.add_argument('--unified', action='store_true', help='Use unified model, which means the masa adapter is built upon the detector model.')
    parser.add_argument('--detector_type', type=str, default='mmdet', help='Choose detector type')
    parser.add_argument('--fp16', action='store_true', help='Activation fp16 mode')
    parser.add_argument('--no-post', action='store_true', help='Do not post-process the results ')
    parser.add_argument('--show_fps', action='store_true', help='Visualize the fps')
    parser.add_argument('--sam_mask', action='store_true', help='Use SAM to generate mask for segmentation tracking')
    parser.add_argument('--sam_path',  type=str, default='saved_models/pretrain_weights/sam_vit_h_4b8939.pth', help='Default path for SAM models')
    parser.add_argument('--sam_type', type=str, default='vit_h', help='Default type for SAM models')
    parser.add_argument(
        '--wait-time',
        type=float,
        default=1,
        help='The interval of show (s), 0 is block')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    assert args.out or args.save_dir, \
        'Please specify at least one output path with "--out" or "--save_dir".'

    if args.unified:
        masa_model = init_masa(args.masa_config, args.masa_checkpoint, device=args.device)
    else:
        det_model = init_detector(args.det_config, args.det_checkpoint, palette='random', device=args.device)
        masa_model = init_masa(args.masa_config, args.masa_checkpoint, device=args.device)
        det_model.cfg.test_dataloader.dataset.pipeline[0].type = 'mmdet.LoadImageFromNDArray'
        test_pipeline = Compose(det_model.cfg.test_dataloader.dataset.pipeline)

    # 画像入力か動画入力かを判定
    if args.image:
        frame = cv2.imread(args.image)  # 画像を読み込む
        frame_idx = 0
        video_len = 1
    else:
        video_reader = mmcv.VideoReader(args.video)  # 動画を読み込む
        frame_idx = 0
        video_len = len(video_reader)

    # 画像 or 各フレームを処理
    if args.image:
        frames = [frame]  # 単一画像のリスト
    else:
        frames = list(video_reader)  # 動画の全フレームをリスト化

    instances_list = []
    for frame in frames:
        if args.unified:
            track_result = inference_masa(masa_model, frame, frame_id=frame_idx,
                                          video_len=video_len, fp16=args.fp16)
        else:
            result = inference_detector(det_model, frame, test_pipeline=test_pipeline, fp16=args.fp16)
            track_result = inference_masa(masa_model, frame, frame_id=frame_idx,
                                          video_len=video_len, det_bboxes=result.pred_instances.bboxes,
                                          det_labels=result.pred_instances.labels, fp16=args.fp16)

        instances_list.append(track_result.to('cpu'))
        frame_idx += 1

    # 画像として保存
    if args.image and args.out:
        processed_frame = visualize_frame(args, visualizer, frame, instances_list[0], frame_idx=0)
        cv2.imwrite(args.out, processed_frame[:, :, ::-1])  # 画像を保存

    print("Processing completed.")




if __name__ == '__main__':
    main()
