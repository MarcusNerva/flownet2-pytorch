import os
import subprocess
import cv2
from utils.flow_utils import flow2img
import numpy as np
import h5py
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='-1')
    parser.add_argument('--thresh_hold', type=int, default=10)
    args = parser.parse_args()
    dataset_name = args.dataset_name
    thresh_hold = args.thresh_hold
    optflow_path = os.path.join('./store', '{}_optflow.hdf5'.format(dataset_name))
    optvideo_dir = './store/optvideos'
    cnt = 0
    if not os.path.exists(optvideo_dir):
        os.makedirs(optvideo_dir)

    print('='*20, 'construct optvideos begin', '='*20)
    with h5py.File(optflow_path, 'r') as f:
        for vid in f.keys():
            if cnt >= thresh_hold: break

            optflows = f[vid][()]
            optflows_len = optflows.shape[0]
            optflows_save_path = os.path.join(optvideo_dir, '{}.avi'.format(vid))
            writer_handle = cv2.VideoWriter(optflows_save_path,
                                            cv2.VideoWriter_fourcc(*'MJPG'),
                                            5, (256, 256))
            for i in range(optflows_len):
                single_flow = optflows[i, ...].transpose(1, 2, 0)
                im = flow2img(single_flow)
                writer_handle.write(im)
            writer_handle.release()
            cnt += 1
    print('=' * 20, 'construct optvideos end', '=' * 20)

