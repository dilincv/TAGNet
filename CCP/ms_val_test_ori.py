import argparse
from scipy import ndimage
import numpy as np
import json
import torch
from torch.utils import data
from networks.ccpnet import Res_Deeplab
from dataset.datasets import CSDataSet
import os
from PIL import Image as PILImage
import torch.nn as nn
from tqdm import tqdm
# from utils.utils import print_settings
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

DATA_LIST_NAME = 'val.lst'
DATA_DIRECTORY = '/home/lindi/qinghong/dataset/cityscapes/'
DATA_LIST_ROOT = './dataset/list/cityscapes/'
IGNORE_LABEL = 255
NUM_CLASSES = 19
RESTORE_FROM = '/media/qhlin/CCP27/iter_44750.pth'
OUTPUT_DIR = './demo'
GPU = '0' #only support single gpu

parser = argparse.ArgumentParser(description="DeepLabLFOV Network")
parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                    help="Path to the directory containing the PASCAL VOC dataset.")
parser.add_argument("--data-list-root", type=str, default=DATA_LIST_ROOT,
                    help="Path to the file listing the images in the dataset.")
parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                    help="The index of the label to ignore during the training.")
parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                    help="Number of classes to predict (including background).")
parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                    help="Where restore model parameters from.")
parser.add_argument("--gpu", type=str, default=GPU,
                    help="choose gpu device.")
parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                    help=".")
parser.add_argument("--data-list-name", type=str, default=DATA_LIST_NAME,
                    help=".")
args = parser.parse_args()

def pad_image(img, target_size):
    """Pad an image up to the target size."""
    rows_missing = target_size[0] - img.shape[2]
    cols_missing = target_size[1] - img.shape[3]
    padded_img = np.pad(img, ((0, 0), (0, 0), (0, rows_missing), (0, cols_missing)), 'constant')
    return padded_img


def predict_whole(net, image, tile_size):
    image = torch.from_numpy(image)
    interp = nn.Upsample(size=tile_size, mode='bilinear', align_corners=True)
    prediction = net(image.cuda())
    if isinstance(prediction, list):
        prediction = prediction[0]
    prediction = interp(prediction).cpu().data[0].numpy().transpose(1,2,0)
    return prediction

def predict_multiscale(net, image, tile_size, scales, classes, flip_evaluation):
    """
    Predict an image by looking at it with different scales.
        We choose the "predict_whole_img" for the image with less than the original input size,
        for the input of larger size, we would choose the cropping method to ensure that GPU memory is enough.
    """
    image = image.data
    N_, C_, H_, W_ = image.shape
    full_probs = np.zeros((H_, W_, classes))  
    for scale in scales:
        scale = float(scale)
        print("Predicting image scaled by %f" % scale)
        scale_image = ndimage.zoom(image, (1.0, 1.0, scale, scale), order=1, prefilter=False)
        scaled_probs = predict_whole(net, scale_image, tile_size)
        if flip_evaluation == True:
            flip_scaled_probs = predict_whole(net, scale_image[:,:,:,::-1].copy(), tile_size)
            scaled_probs = 0.5 * (scaled_probs + flip_scaled_probs[:,::-1,:])
        full_probs += scaled_probs
    full_probs /= len(scales)
    return full_probs

def get_confusion_matrix(gt_label, pred_label, class_num):
        """
        Calcute the confusion matrix by given label and pred
        :param gt_label: the ground truth label
        :param pred_label: the pred label
        :param class_num: the nunber of class
        :return: the confusion matrix
        """
        index = (gt_label * class_num + pred_label).astype('int32')
        label_count = np.bincount(index)
        confusion_matrix = np.zeros((class_num, class_num))

        for i_label in range(class_num):
            for i_pred_label in range(class_num):
                cur_index = i_label * class_num + i_pred_label
                if cur_index < len(label_count):
                    confusion_matrix[i_label, i_pred_label] = label_count[cur_index]

        return confusion_matrix

def train_id_to_label_id(result):
    result = torch.from_numpy(result)
    result = result.long()
    H, W = result.shape
    label_id = torch.LongTensor([7,8,11,12,13,17,19,20,21,22,23,24,25,26,27,28,31,32,33])
    result = result.reshape(H*W)
    result = torch.gather(label_id, 0, result)
    result = result.reshape(H,W)
    result = result.numpy()
    result = result.astype(np.uint8)
    return result

def main():
    """Create the model and start the evaluation process."""
    # print_settings(args.__dict__, 'ss_val_test_multigpu')
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.data_list_name == 'test.lst' and len(os.listdir(args.output_dir)) != 0:
        print('=======!!!THE TEST OUTPUT DIR IS NOT EMPTY!!!=======')
        return -1
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    input_size = (1024, 2048)

    model = Res_Deeplab(num_classes=args.num_classes)
    
    saved_state_dict = torch.load(args.restore_from)
    model.load_state_dict(saved_state_dict)

    model.eval()
    model.cuda()

    testloader = data.DataLoader(CSDataSet(args.data_dir, os.path.join(args.data_list_root,args.data_list_name), crop_size=(1024, 2048),
                                           mean=IMG_MEAN, scale=False, mirror=False, max_iters=None, ignore_label=args.ignore_label),
                                    batch_size=1, shuffle=False, pin_memory=True)

    confusion_matrix = np.zeros((args.num_classes,args.num_classes))

    pbar = tqdm(enumerate(testloader), total=len(testloader))
    for index, batch in pbar:
        if index % 100 == 0:
            print('%d processd'%(index))
        image, label, size, name = batch
        size = size[0].numpy()
        with torch.no_grad():
            output = predict_multiscale(model, image, input_size, [0.75, 1.0, 1.25, 1.5, 1.75, 2.0], args.num_classes, True)
        seg_pred = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
        output_im = PILImage.fromarray(train_id_to_label_id(seg_pred))
        output_im.save(os.path.join(args.output_dir, name[0] + '.png'))

        seg_gt = np.asarray(label[0].numpy()[:size[0],:size[1]], dtype=np.int)
    
        ignore_index = seg_gt != 255
        seg_gt = seg_gt[ignore_index]
        seg_pred = seg_pred[ignore_index]
        confusion_matrix += get_confusion_matrix(seg_gt, seg_pred, args.num_classes)

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)

    IU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IU = IU_array.mean()
    pbar.set_description('[VAL]')
    # getConfusionMatrixPlot(confusion_matrix)
    print({'meanIU':mean_IU, 'IU_array':IU_array})
    with open('result.txt', 'w') as f:
        f.write(json.dumps({'meanIU':mean_IU, 'IU_array':IU_array.tolist()}))

if __name__ == '__main__':
    main()
