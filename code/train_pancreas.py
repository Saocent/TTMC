import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss, MSELoss, L1Loss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

# from networks.discriminator import FC3DDiscriminator
import test_util
# from dataloaders.la_heart import LAHeart, RandomCrop, ToTensor, TwoStreamBatchSampler, SingleStreamBatchSampler
from dataloaders.pancreas import Pancreas, RandomCrop, ToTensor, TwoStreamBatchSampler
from networks.vnet_TTMC import VNet
# from networks.vnet_sdf_MC_Srecon import VNet
from utils import ramps, losses, metrics
from utils.util import compute_sdf

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/pancreas/', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='Pancreas/TTMC_6_56_1400', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=6000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=2,
                    help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float, default=0.01,
                    help='maximum epoch number to train')
parser.add_argument('--D_lr', type=float, default=1e-4,
                    help='maximum discriminator learning rate to train')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--labelnum', type=int, default=12, help='random seed')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--consistency_weight', type=float, default=0.1,
                    help='balance factor to control supervised loss and consistency loss')
parser.add_argument('--gpu', type=str, default='1', help='GPU to use')
parser.add_argument('--alpha', type=float, default=0.4,
                    help='balance factor to control regional and sdm loss')
parser.add_argument('--beta', type=float, default=0.3,
                    help='balance factor to control regional and sdm loss')
parser.add_argument('--gamma', type=float, default=0.3,
                    help='balance factor to control supervised and consistency loss')
# costs
parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="kl", help='consistency_type')
parser.add_argument('--with_cons', type=str,
                    default="without_cons", help='with or without consistency')
parser.add_argument('--consistency', type=float,
                    default=1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=40.0, help='consistency_rampup')
args = parser.parse_args()

train_data_path = args.root_path
snapshot_path = "../model/" + args.exp + \
                "_{}labels_alpha_{}_beta_{}_gamma_{}_conweight_{}/".format(
                    args.labelnum, args.alpha, args.beta, args.gamma, args.consistency)

# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs

if not args.deterministic:
    cudnn.benchmark = True
    cudnn.deterministic = False
else:
    cudnn.benchmark = False  # True #
    cudnn.deterministic = True  # False #
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

num_classes = 2
patch_size = (96, 96, 128)


# (112, 112, 80) (160, 128, 80)
# (96, 80, 160)

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


if __name__ == "__main__":
    # make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))


    def create_model(ema=False):
        # Network definition
        net = VNet(n_channels=1, n_classes=num_classes - 1, normalization='batchnorm', has_dropout=True,
                   has_skipconnect=False)
        model = net.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model


    model = create_model()

    db_train = Pancreas(base_dir=train_data_path,
                        split='train',  # train/val split
                        transform=transforms.Compose([
                            RandomCrop(patch_size),
                            ToTensor(),
                        ]))

    labelnum = args.labelnum  # default 16
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, 62))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size - labeled_bs)


    # batch_sampler = SingleStreamBatchSampler(labeled_idxs, batch_size)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)


    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    model.train()

    optimizer = optim.Adam(model.parameters(), lr=base_lr, betas=(0.9, 0.99), amsgrad=False)
    ce_loss = BCEWithLogitsLoss()
    mse_loss = MSELoss()
    l1_loss = L1Loss()

    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))
    best_dice = 0
    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    lr_ = base_lr
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=50)
    for epoch_num in iterator:
        time1 = time.time()
        for i_batch, sampled_batch in enumerate(trainloader):

            time2 = time.time()
            # print('fetch data cost {}'.format(time2-time1))
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            outputs_seg, outputs_tanh, outputs_Sre = model(volume_batch)

            # calculate the loss
            with torch.no_grad():
                gt_sdm = compute_sdf(label_batch[:].cpu().numpy(), outputs_seg[:labeled_bs, 0, ...].shape)
                # gt_Tsdm = -gt_sdm + numpy.sign(gt_sdm)
                gt_Sre = numpy.tanh(numpy.multiply(gt_sdm, volume_batch[:labeled_bs, 0, ...].cpu().numpy()))
                gt_sdm = torch.from_numpy(gt_sdm).float().cuda()
                gt_Sre = torch.from_numpy(gt_Sre).float().cuda()
                gt_softmask = torch.sigmoid(-150 * gt_sdm)

            loss_sdfl2 = mse_loss(outputs_tanh[:labeled_bs, 0, ...], gt_sdm)
            loss_Sre = args.gamma * mse_loss(outputs_Sre[:labeled_bs, 0, ...], gt_Sre)
            loss_seg = ce_loss(
                outputs_seg[:labeled_bs, 0, ...], label_batch[:labeled_bs].float())
            loss_seg_dice = args.alpha * losses.dice_loss(
                outputs_seg[:labeled_bs, 0, :, :, :], label_batch[:labeled_bs])

            sdm_to_mask = torch.sigmoid(-150 * outputs_tanh)
            # sdm_to_Tsdm = -outputs_tanh + torch.sign(outputs_tanh)
            Tsdm_to_Sre = torch.tanh(torch.mul(outputs_tanh, volume_batch))

            loss_sdfdice = losses.dice_loss(sdm_to_mask[:labeled_bs, 0, ...], gt_softmask)
            loss_sdf = args.beta * loss_sdfl2
            consistency_loss = mse_loss(sdm_to_mask, outputs_seg) + mse_loss(Tsdm_to_Sre, outputs_Sre)
            # consistency_loss = ce_loss(outputs, sdm_to_mask)
            supervised_loss = loss_seg_dice + loss_sdf + loss_Sre

            #consistency_weight = get_current_consistency_weight(iter_num // 150)
            consistency_weight = get_current_consistency_weight(np.clip(iter_num, 0, 1400) // 150)

            loss = supervised_loss + consistency_weight * consistency_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            dc = metrics.dice(torch.argmax(
                outputs_seg[:labeled_bs], dim=1), label_batch[:labeled_bs])

            iter_num = iter_num + 1
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)
            writer.add_scalar('loss/loss_Sre', loss_Sre, iter_num)
            writer.add_scalar('loss/loss_dice', loss_seg_dice, iter_num)
            writer.add_scalar('loss/loss_hausdorff', loss_sdf, iter_num)
            writer.add_scalar('loss/consistency_weight',
                              consistency_weight, iter_num)
            writer.add_scalar('loss/consistency_loss',
                              consistency_loss, iter_num)

            logging.info(
                'iteration %d : loss : %f, loss_consis: %f, loss_sdf: %f, loss_Sre: %f, loss_dice: %f, consis_weight: %f' %
                (iter_num, loss.item(), consistency_loss.item(), loss_sdf.item(),
                 loss_Sre.item(), loss_seg_dice.item(), consistency_weight))
            writer.add_scalar('loss/loss', loss, iter_num)
            logging.info('iteration %d : loss : %f' % (iter_num, loss.item()))
            if iter_num % 50 == 0:
                image = volume_batch[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/Image', grid_image, iter_num)

                image = outputs_seg[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Predicted_label', grid_image, iter_num)

                image = sdm_to_mask[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/sdm2Mask', grid_image, iter_num)

                image = outputs_tanh[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/sdmtMap', grid_image, iter_num)

                image = label_batch[0, :, :, 20:61:10].unsqueeze(
                    0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Groundtruth_label',
                                 grid_image, iter_num)

                image = gt_sdm[0, :, :, 20:61:10].unsqueeze(
                    0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Groundtruth_sdmtMap',
                                 grid_image, iter_num)

            # change lr
            if iter_num % 2500 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            if iter_num >= 800 / (args.labeled_bs / 2) and iter_num % (200 / (args.labeled_bs / 2)) == 0:
                model.eval()
                dice_sample = test_util.var_all_case(model, num_classes=num_classes, patch_size=patch_size,
                                                     stride_xy=16, stride_z=16, data_path=args.root_path, nms=1)

                if dice_sample > best_dice:
                    best_dice = dice_sample
                    save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, best_dice))
                    save_best_path = os.path.join(snapshot_path, 'best_model.pth')
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)
                    logging.info("save best model to {}".format(save_mode_path))
                writer.add_scalar('Var_dice/Dice', dice_sample, iter_num)
                writer.add_scalar('Var_dice/Best_dice', best_dice, iter_num)
                model.train()

            if iter_num >= max_iterations:
                save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
