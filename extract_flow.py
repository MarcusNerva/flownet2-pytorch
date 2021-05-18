import torch
import glob
import numpy as np
import cv2
from utils.frame_utils import resize_frame
from models import FlowNet2
import argparse
import os
import h5py
import tqdm


def get_frames(video_path, stride):
    try:
        cap = cv2.VideoCapture(video_path)
    except:
        print('Can\'t open video {}'.format(video_path))
        return None

    frames, frame_cnt = [], 0
    while True:
        ret, frame = cap.read()
        if ret is False: break
        frame = resize_frame(frame)
        frame = frame[:, :, ::-1]  # BGR2RGB
        frames.append(frame)
        frame_cnt += 1

    indices = list(range(8, frame_cnt - 7, stride))
    ret = []
    for idx in indices:
        ret.append([frames[idx], frames[idx + 1]])  # adjacent two frames
    return ret


if __name__ == '__main__':
    print('='*20, 'extract optflow start', '='*20)
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', type=str, default='-1')
    parser.add_argument('--model_name', type=str, default='-1')
    parser.add_argument('--video_dir', type=str, default='-1')
    parser.add_argument('--ext', type=str, default='-1')
    parser.add_argument('--stride', type=int, default=5)
    parser.add_argument('--save_dir', type=str, default='./store')
    parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument("--rgb_max", type=float, default=255.)
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    dataset_name = args.dataset_name
    model_name = args.model_name
    video_dir = args.video_dir
    ext = args.ext
    stride = args.stride
    save_dir = args.save_dir
    save_path = os.path.join(save_dir, '{}_optflow.hdf5'.format(dataset_name))
    model_dict_path = './checkpoints/{}_checkpoint.pth.tar'.format(model_name)
    assert video_dir != '-1', 'please set arguments video_dir'
    assert save_dir != '-1', 'please set argument save_dir'
    assert ext != '-1', 'please set argument ext'
    assert model_name != '-1', 'please set argument model_name'
    if not os.path.exists('./checkpoints'):
        os.makedirs('./checkpoints')
        print('#'*20, 'checkpoints directory is created!', '#' * 20)

    # initialize FlowNet
    if not os.path.exists(model_dict_path):
        raise AssertionError('{}_checkpoint.pth.tar is not downloaded!'.format(model_name))
    net = FlowNet2(args).cuda()
    dict = torch.load(model_dict_path)
    net.load_state_dict(dict['state_dict'])

    # prepare videos
    video_list = glob.glob(os.path.join(video_dir, '*.{}'.format(ext)))

    # begin extraction
    with h5py.File(save_path, 'w') as f:
        for video_path in tqdm.tqdm(video_list):
            video_id = os.path.basename(video_path)
            video_id = video_id.split('.')[0]
            frames_set = get_frames(video_path, stride=stride)
            optical_flow_store = []

            for item in frames_set:
                im0, im1 = item[0], item[1]
                images = [im0, im1]
                images = np.array(images).transpose(3, 0, 1, 2)
                im = torch.from_numpy(images.astype(np.float32)).unsqueeze(0).cuda()
                result = net(im)  # (1, H, W, 2)
                result = result.detach().cpu().numpy()
                optical_flow_store.append(result)
            f[video_id] = np.concatenate(optical_flow_store, axis=0)  # (len(frames_set), H, W, 2)

    print('='*20, 'finished', '='*20)

