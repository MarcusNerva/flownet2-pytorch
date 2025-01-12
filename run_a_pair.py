import torch
import numpy as np
import argparse
import cv2
from utils.frame_utils import resize_frame
from utils.flow_utils import flow2img
import PIL.Image as Image

from models import FlowNet2  # the path is depended on where you create this module
from utils.frame_utils import read_gen  # the path is depended on where you create this module
import imageio


if __name__ == '__main__':
    # obtain the necessary args for construct the flownet framework
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument("--rgb_max", type=float, default=255.)
    
    args = parser.parse_args()

    # initial a Net
    net = FlowNet2(args).cuda()
    # load the state_dict
    dict = torch.load("./checkpoints/FlowNet2_checkpoint.pth.tar")
    net.load_state_dict(dict["state_dict"])

    # load the image pair, you can find this operation in dataset.py
    pim1 = cv2.imread("./store/realcat.jpg")
    pim2 = cv2.imread("./store/realcat.jpg")
    pim1, pim2 = resize_frame(pim1), resize_frame(pim2)
    pim1, pim2 = pim1[:, :, ::-1], pim2[:, :, ::-1]
    images = [pim1, pim2]
    images = np.array(images).transpose(3, 0, 1, 2)
    im = torch.from_numpy(images.astype(np.float32)).unsqueeze(0).cuda()
    # im = torch.cat([im, im], dim=0)

    # process the image pair to obtian the flow
    print(net._get_name())
    # print(im.shape)
    print('im.shape:{}'.format(im.shape))
    result = net(im).squeeze()
    print('result.shape:{}'.format(result.shape))


    # save flow, I reference the code in scripts/run-flownet.py in flownet2-caffe project
    def writeFlow(name, flow):
        f = open(name, 'wb')
        f.write('PIEH'.encode('utf-8'))
        np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
        flow = flow.astype(np.float32)
        flow.tofile(f)
        f.flush()
        f.close()


    data = result.data.cpu().numpy().transpose(1, 2, 0)
    print(data.shape)
    temp = flow2img(data)
    temp = Image.fromarray(temp)
    temp.show()
    # writeFlow("/home/hjj/flownet2-master/data/FlyingChairs_examples/0000007-img.flo", data)
