import argparse
import os

import torch

from networks.vnet_TTMC import VNet
from test_util import test_all_case

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../../test/data/pancreas/pancreas96/', help='Name of Experiment')
parser.add_argument('--model', type=str,
                    default='Pancreas/TTMC_12_50_iter1700_12labels_alpha_0.4_beta_0.3_gamma_0.3_conweight_1/',
                    help='model_name')
parser.add_argument('--gpu', type=str, default='1', help='GPU to use')
parser.add_argument('--detail', type=int, default=1,
                    help='print metrics for every samples?')
parser.add_argument('--nms', type=int, default=0,
                    help='apply NMS post-procssing?')

FLAGS = parser.parse_args()

# os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
snapshot_path = "../model/{}".format(FLAGS.model)

num_classes = 2

test_save_path = os.path.join(snapshot_path, "test/")
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)
print(test_save_path)
with open(FLAGS.root_path + '/test.txt', 'r') as f:
    image_list = f.readlines()
image_list = [FLAGS.root_path + "/" + item.replace('\n', '') + ".h5" for item in
              image_list]


# has_skipconnect=False
def test_calculate_metric():
    net = VNet(n_channels=1, n_classes=num_classes - 1, normalization='batchnorm', has_dropout=False,
               has_skipconnect=False).cuda()
    save_mode_path = os.path.join(snapshot_path, 'best_model.pth')
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()

    avg_metric = test_all_case(net, image_list, num_classes=num_classes,
                               patch_size=(96, 96, 128), stride_xy=16, stride_z=16,
                               save_result=True, test_save_path=test_save_path,
                               metric_detail=FLAGS.detail, nms=FLAGS.nms)

    return avg_metric


if __name__ == '__main__':
    metric = test_calculate_metric()  # 6000
    # print(metric)
