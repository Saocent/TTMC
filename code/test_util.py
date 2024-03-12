import math

import h5py
import nibabel as nib
import numpy
import numpy as np
import torch
from medpy import metric
from skimage.measure import label
from tqdm import tqdm


def getLargestCC(segmentation):
    labels = label(segmentation)
    if labels.max() == 0:  # assume at least 1 CC
        print('no connected regions')
        print("-" * 100)
        print("-" * 100)
        return segmentation
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largestCC


def var_all_case(model, num_classes, patch_size=(112, 112, 80), stride_xy=18, stride_z=4, data_path='', nms=0):
    # with open(data_path + '/test.list', 'r') as f:
    #     image_list = f.readlines()
    # image_list = [data_path + item.replace('\n', '') + "/mri_norm2.h5" for item in
    #               image_list]
    with open(data_path + '/test.txt', 'r') as f:
        image_list = f.readlines()
    image_list = [data_path + "/" + item.replace('\n', '') + ".h5" for item in image_list]

    loader = tqdm(image_list)
    total_dice_seg = 0.0
    total_dice_sdf = 0.0
    for image_path in loader:
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]


        prediction_seg, prediction_sdf = test_single_case(model, image, stride_xy, stride_z, patch_size, num_classes=num_classes)


        if nms:
            prediction_seg = getLargestCC(prediction_seg)
            prediction_sdf = getLargestCC(prediction_sdf)
        if np.sum(prediction_seg) == 0:
            dice_seg = 0
        else:
            dice_seg = metric.binary.dc(prediction_seg, label[:])
        if np.sum(prediction_sdf) == 0:
            dice_sdf = 0
        else:
            dice_sdf = metric.binary.dc(prediction_sdf, label[:])
        total_dice_seg += dice_seg
        total_dice_sdf += dice_sdf

    # avg_dice = max(total_dice_seg, total_dice_sdf) / len(image_list)
    avg_dice = total_dice_seg / len(image_list)
    print('average metric is {}'.format(avg_dice))
    return avg_dice


def test_all_case(net, image_list, num_classes, patch_size=(352, 192, 192), stride_xy=18, stride_z=4, save_result=True,
                  test_save_path=None, preproc_fn=None, metric_detail=0, nms=0):
    total_metric_seg = 0.0
    total_metric_sdf = 0.0
    loader = tqdm(image_list) if not metric_detail else image_list
    ith = 0
    for image_path in loader:
        # id = image_path.split('/')[-2]
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        if preproc_fn is not None:
            image = preproc_fn(image)
        prediction_seg, prediction_sdf = test_single_case(
            net, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
        # prediction_seg, prediction_sdf = test_single_case(
        #     net, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
        # if nms:
        #     prediction_seg = getLargestCC(prediction_seg)
        #     prediction_sdf = getLargestCC(prediction_sdf)

        if np.sum(prediction_seg) == 0:
            single_metric_seg = (0, 0, 0, 0)
            single_metric_sdf = (0, 0, 0, 0)
        else:
            single_metric_seg = calculate_metric_percase(prediction_seg, label[:])
            single_metric_sdf = calculate_metric_percase(prediction_sdf, label[:])
        if metric_detail:
            print('%02d,\t%.5f, %.5f, %.5f, %.5f' % (
                ith, single_metric_seg[0], single_metric_seg[1], single_metric_seg[2], single_metric_seg[3]))
            # print('%02d,\t%.5f, %.5f, %.5f, %.5f' % (
            #     ith, single_metric_sdf[0], single_metric_sdf[1], single_metric_sdf[2], single_metric_sdf[3]))

        total_metric_seg += np.asarray(single_metric_seg)
        total_metric_sdf += np.asarray(single_metric_sdf)

        # print(ith)
        if save_result:
            nib.save(nib.Nifti1Image(prediction_seg.astype(np.float32),
                                     np.eye(4)), test_save_path + "%02d_pred_seg.nii.gz" % ith)
            # nib.save(nib.Nifti1Image(prediction_seg.astype(np.float32),
            #                          np.eye(4)), test_save_path + "%02d_score_seg.nii.gz" % ith)
            # nib.save(nib.Nifti1Image(prediction_sdf.astype(np.float32),
            #                          np.eye(4)), test_save_path + "%02d_pred_sdf.nii.gz" % ith)
            # nib.save(nib.Nifti1Image(prediction_sre.astype(np.float32),
            #                          np.eye(4)), test_save_path + "%02d_pred_sre.nii.gz" % ith)
            # nib.save(nib.Nifti1Image(pre_conseg.astype(np.float32),
            #                          np.eye(4)), test_save_path + "%02d_pred_conseg.nii.gz" % ith)
            # nib.save(nib.Nifti1Image(pre_consre.astype(np.float32),
            #                          np.eye(4)), test_save_path + "%02d_pred_consre.nii.gz" % ith)
            # nib.save(nib.Nifti1Image(image[:].astype(np.float32), np.eye(
            #     4)), test_save_path + "%02d_img.nii.gz" % ith)
            # nib.save(nib.Nifti1Image(label[:].astype(np.float32), np.eye(
            #     4)), test_save_path + "%02d_gt.nii.gz" % ith)
        ith += 1

    avg_metric_seg = total_metric_seg / len(image_list)
    avg_metric_sdf = total_metric_sdf / len(image_list)
    print('average metric is {}'.format(avg_metric_seg))
    print('average metric is {}'.format(avg_metric_sdf))

    return avg_metric_seg


def test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=1):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0] - w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1] - h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2] - d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
    hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2
    dl_pad, dr_pad = d_pad // 2, d_pad - d_pad // 2
    if add_pad:
        image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad),
                               (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map_seg = np.zeros((num_classes,) + image.shape).astype(np.float32)
    score_map_sdf = np.zeros((num_classes,) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy * x, ww - patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh - patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd - patch_size[2])
                test_patch = image[xs:xs + patch_size[0],
                             ys:ys + patch_size[1], zs:zs + patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(
                    test_patch, axis=0), axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                with torch.no_grad():
                    y, y1_tanh, y2 = net(test_patch)
                    # ensemble
                    dis_to_mask = torch.sigmoid(-1500 * y1_tanh)

                y = y.cpu().data.numpy()
                dis2mask = dis_to_mask.cpu().data.numpy()
                y = y[0, :, :, :, :]
                dis2mask = dis2mask[0, :, :, :, :]
                score_map_seg[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                    = score_map_seg[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + y
                score_map_sdf[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                    = score_map_sdf[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + dis2mask
                cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                    = cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + 1
    score_map_seg = score_map_seg / np.expand_dims(cnt, axis=0)
    score_map_sdf = score_map_sdf / np.expand_dims(cnt, axis=0)
    label_map_seg = (score_map_seg[0] > 0.5).astype(np.int)
    label_map_sdf = (score_map_sdf[0] > 0.5).astype(np.int)

    if add_pad:
        label_map_seg = label_map_seg[wl_pad:wl_pad + w,
                        hl_pad:hl_pad + h, dl_pad:dl_pad + d]
        label_map_sdf = label_map_sdf[wl_pad:wl_pad + w,
                        hl_pad:hl_pad + h, dl_pad:dl_pad + d]
        score_map_seg = score_map_seg[:, wl_pad:wl_pad +
                                                w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
        score_map_sdf = score_map_sdf[:, wl_pad:wl_pad +
                                                w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
    return label_map_seg, label_map_sdf


def cal_dice(prediction, label, num=2):
    total_dice = np.zeros(num - 1)
    for i in range(1, num):
        prediction_tmp = (prediction == i)
        label_tmp = (label == i)
        prediction_tmp = prediction_tmp.astype(np.float)
        label_tmp = label_tmp.astype(np.float)

        dice = 2 * np.sum(prediction_tmp * label_tmp) / \
               (np.sum(prediction_tmp) + np.sum(label_tmp))
        total_dice[i - 1] += dice

    return total_dice


def calculate_metric_percase(pred, gt):
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)

    return dice, jc, hd, asd
