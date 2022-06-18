from torch.utils import data
from dataset.datasets import CSDataSet_vis
import numpy as np
import torch
import cv2
import os
import os.path as osp

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

COLORS = [[0, 0, 255], [255, 255, 255], [225, 105, 65], [0, 97, 255], [205, 90, 106], [15, 94, 56], [0, 255, 0],[64, 145, 61], [36, 51, 135],
          [0, 0, 0], [135, 138, 128], [18, 153, 255], [105, 128, 112], [230, 224, 176], [105, 165, 218]]


def get_output_dir_rand(snapshot_dir, i_iter):
    output_dir_rand = osp.join(snapshot_dir, 'vis', str(i_iter))
    if not os.path.exists(output_dir_rand):
        os.makedirs(output_dir_rand)
    return output_dir_rand


def get_output_dir_spe(snapshot_dir, index):
    output_dir_spe = osp.join(snapshot_dir, 'vis', 'spe', str(index))
    if not os.path.exists(output_dir_spe):
        os.makedirs(output_dir_spe)
    return output_dir_spe


def process_vis_variables(blocks_center, rois_center, rois, scale):
    blocks_center = torch.stack(blocks_center, dim=-1) * scale
    for i in range(len(rois_center)):
        rois_center[i] = torch.stack(rois_center[i], dim=-1)[0] * scale
    for i in range(len(rois)):
        rois[i] = rois[i][0, :, :, :, 1:] * scale
    return blocks_center, rois_center, rois


def process_image(image, img_mean):
    image = np.asarray(image[0])
    image = np.transpose(image, [1, 2, 0])
    image = image + img_mean
    image = image.astype(np.uint8)
    return image


def plot(image, blocks_center, rois_center, rois, locations, scale, colors):
    assert len(colors) >= len(locations)
    image = process_image(image, IMG_MEAN)
    blocks_center, rois_center, rois = process_vis_variables(blocks_center, rois_center, rois, scale)

    for location, color in zip(locations, colors):
        case = get_one_case(blocks_center, rois_center, rois, location)
        image = plot_one_case(image, case, color=color)

    return image


def get_one_case(blocks_center, rois_center, rois, location):
    x, y = location
    case = {'center_points': [], 'rois': []}

    case['center_points'].append(blocks_center[:, x, y, :])
    for roi_center in rois_center:
        case['center_points'].append(roi_center[:, x, y, :])
    for roi in rois:
        case['rois'].append(roi[:, x, y, :])

    case['rois'] = torch.cat(case['rois'], dim=0)
    return case


def plot_one_case(image, case, color):
    image = image.copy()
    rois = case['rois']
    center_points = case['center_points']
    for roi in rois:
        cv2.rectangle(image, (roi[0], roi[1]), (roi[2], roi[3]), color, thickness=2)
    for i in range(len(center_points) - 1):
        for j in range(center_points[0].shape[0]):
            cv2.line(image, tuple(center_points[i][j, :].int().tolist()),
                     tuple(center_points[i+1][j, :].int().tolist()), color, thickness=2)

    cv2.circle(image, tuple(center_points[0][0, :].int().tolist()), 3, color, thickness=3)

    return image


def vis_output(train_args, model, dataset, i_iter):
    print('doing visualization...')
    gpu = train_args.gpu.split(',')[0]
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    model = model.cuda()

    with torch.no_grad():
        model.eval()
        testloader = data.DataLoader(dataset, batch_size=1, shuffle=True, pin_memory=True)
        for index, batch in enumerate(testloader):
            if index == train_args.vis_rand_n_image:
                break
            image, label, size, name = batch
            model(image.cuda(), vis=True)

            rois = model.ccp.variables_for_vis['rois']
            rois_center = model.ccp.variables_for_vis['rois_center']
            blocks_center = model.ccp.blocks_center
            feature_size_with_block = model.ccp.feature_size_with_block
            scale = image.shape[2] / model.ccp.input_size[0]
            rand_locations = np.random.randint(0, feature_size_with_block, size=[train_args.vis_rand_n_point, 2])
            image = plot(image, blocks_center, rois_center, rois, rand_locations, scale, COLORS)
            output_dir = get_output_dir_rand(train_args.snapshot_dir, i_iter)
            cv2.imwrite(osp.join(output_dir, name[0] + '.png'), image)

        for i in range(len(train_args.vis_spe_images)):
            image, label, size, name = dataset.__getitem__(train_args.vis_spe_images[i])
            image = image.unsqueeze(0)
            model(image.cuda(), vis=True)

            rois = model.ccp.variables_for_vis['rois']
            rois_center = model.ccp.variables_for_vis['rois_center']
            blocks_center = model.ccp.blocks_center
            scale = image.shape[2] / model.ccp.input_size[0]
            spe_locations = train_args.vis_spe_points

            image = plot(image, blocks_center, rois_center, rois, spe_locations, scale, COLORS)
            output_dir = get_output_dir_spe(train_args.snapshot_dir, i)
            cv2.imwrite(osp.join(output_dir, name + '_iter_' + str(i_iter) + '.png'), image)

    model.train()
    os.environ["CUDA_VISIBLE_DEVICES"] = train_args.gpu


if __name__ == '__main__':
    from networks.ccpnet import Res_Deeplab
    import argparse
    import shutil

    DATA_DIR = '/home/xieke/wuyong/data/datasets/cityscapes'
    DATA_LIST = './dataset/list/cityscapes/val.lst'
    IGNORE_LABEL = 255
    INPUT_SIZE = (769, 769)
    NUM_CLASSES = 19
    RESTORE_ROOT = '/media/szu/mydata/wuyong/snapshots/ccp_3'
    MODELS_NAME = ['iter_0.pth']
    SNAPSHOT_DIR = './debug'
    GPU = '4'
    VIS_RAND_N_IMAGE = 7
    VIS_RAND_N_POINT = 7
    VIS_SPE_IMAGES = (0, 1, 2, 3, 4)
    VIS_SPE_POINTS = ((12, 11), (32, 19), (15, 20), (21, 29), (2, 8), (24, 31), (12, 22))


    parser = argparse.ArgumentParser(description="CCP Network")
    parser.add_argument("--data-dir", type=str, default=DATA_DIR,
                        help="Path to the directory containing the cs dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=tuple, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--gpu", type=str, default=GPU,
                        help="choose gpu device.")
    parser.add_argument("--vis_rand-n-image", type=int, default=VIS_RAND_N_IMAGE,
                        help="")
    parser.add_argument("--vis_rand-n-point", type=int, default=VIS_RAND_N_POINT,
                        help="")
    parser.add_argument("--vis-spe-images", type=tuple, default=VIS_SPE_IMAGES,
                        help="")
    parser.add_argument("--vis_spe_points", type=tuple, default=VIS_SPE_POINTS,
                        help="")

    args = parser.parse_args()

    model = Res_Deeplab(num_classes=NUM_CLASSES)
    dataset_vis = CSDataSet_vis(root=args.data_dir, list_path=args.data_list, max_iters=None,
                                crop_size=args.input_size, mean=IMG_MEAN, scale=False, mirror=False, ignore_label=args.ignore_label)
    if os.path.exists(os.path.join(args.snapshot_dir, 'vis')):
        shutil.rmtree(os.path.join(args.snapshot_dir, 'vis'))
    for model_name in MODELS_NAME:
        saved_state_dict = torch.load(os.path.join(RESTORE_ROOT, model_name))
        model.load_state_dict(saved_state_dict)
        i_iter = model_name.split('.')[0].split('_')[-1]
        vis_output(args, model, dataset_vis, i_iter)
    print('================Finish!!!================')
