import argparse
import numpy as np
import torch
from torch.utils import data
from networks.ccpnet import Res_Deeplab
import os
from tqdm import tqdm
from PIL import Image as PILImage
from utils.yfms_utils import MultiEvalModule, test_batchify_fn
from dataset.datasets import CSYFMSDataSet
from utils.utils import print_settings
from utils.utils import train_id_to_label_id
from utils.utils import get_confusion_matrix
from ensemble_multicpu import save_score_npy
import torch.nn.functional as F

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

DATA_LIST_NAME = 'val.lst'
DATA_DIRECTORY = '/media/szu/mydata/wuyong/datasets/cityscapes'
DATA_LIST_ROOT = './dataset/list/cityscapes/'
IGNORE_LABEL = 255
NUM_CLASSES = 19
RESTORE_FROM = '/home/xieke/wuyong/codes/CCP/CCP35/snapshot/iter_44750.pth'
OUTPUT_DIR = './ms_val_outputs_yf'
GPU = '0,1,2,3,4,5,6,7'
INPUT_SIZE = '769,769'
SAVE_NPY_DIR = None


parser = argparse.ArgumentParser(description="DeepLabLFOV Network")
parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                    help="Path to the directory containing the PASCAL VOC dataset.")
parser.add_argument("--data-list-root", type=str, default=DATA_LIST_ROOT,
                    help="Path to the file listing the images in the dataset.")
parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                    help="The index of the label to ignore during the training.")
parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                    help="Number of classes to predict (including background).")
parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                    help="Where restore model parameters from.")
parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                    help="Comma-separated string with height and width of images.")
parser.add_argument("--data-list-name", type=str, default=DATA_LIST_NAME,
                    help=".")
parser.add_argument("--gpu", type=str, default=GPU,
                    help="choose gpu device.")
parser.add_argument("--save_npy_dir", type=str, default=SAVE_NPY_DIR,
                    help="")

args = parser.parse_args()


def eval_batch(images, labels, filenames, evaluator):
    outputs = evaluator.parallel_forward(images)
    confusion_matrix = 0
    for output, label, filename in zip(outputs, labels, filenames):
        output= torch.Tensor.squeeze(output, dim=0)
        output = output.cpu()
        if args.save_npy_dir is not None:
            output_sum = output.sum(dim=[0], keepdim=True)
            output_npy = output/output_sum
            save_score_npy(output_npy.numpy(),
                           os.path.join(args.save_npy_dir, filename.split('/')[-1].split('.')[0]))

        pred = np.asarray(np.argmax(output, axis=0), dtype=np.uint8)

        output_im = PILImage.fromarray(train_id_to_label_id(pred))
        output_im.save(os.path.join(args.output_dir, filename.split('/')[3]))

        seg_gt = np.asarray(label, dtype=np.int)
        ignore_index = seg_gt != args.ignore_label
        seg_gt = seg_gt[ignore_index]
        pred = pred[ignore_index]

        confusion_matrix += get_confusion_matrix(seg_gt, pred, args.num_classes)

    return confusion_matrix


def main():
    """Create the model and start the evaluation process."""
    print_settings(args.__dict__, 'ms_val_test_yf')
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)
    dataset = CSYFMSDataSet(root=args.data_dir, data_list=os.path.join(args.data_list_root,args.data_list_name), mean=IMG_MEAN,
                            ignore_label=args.ignore_label, crop_size=input_size)
    testloader = data.DataLoader(
        dataset,
        batch_size=len(args.gpu.split(',')),
        num_workers=len(args.gpu.split(',')),
        shuffle=False,
        collate_fn=test_batchify_fn
    )
    model = Res_Deeplab(num_classes=args.num_classes)
    saved_state_dict = torch.load(args.restore_from)
    model.load_state_dict(saved_state_dict)
    print("loading the cheickpoint from the path{}".format(args.restore_from))

    evaluator = MultiEvalModule(model, args.num_classes).cuda()
    evaluator.eval()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.save_npy_dir is not None and not os.path.exists(args.save_npy_dir):
        os.makedirs(args.save_npy_dir)
    if args.data_list_name == 'test.lst' and len(os.listdir(args.output_dir)) != 0:
        print('=======!!!THE TEST OUTPUT DIR IS NOT EMPTY!!!=======')
        return -1
    tbar = tqdm(testloader)
    with torch.no_grad():
        confusion_matrix_total = 0
        for index, (images, labels, filenames) in enumerate(tbar):
            confusion_matrix = eval_batch(images, labels, filenames, evaluator)
            confusion_matrix_total = confusion_matrix_total + confusion_matrix
        pos = confusion_matrix_total.sum(1)
        res = confusion_matrix_total.sum(0)
        tp = np.diag(confusion_matrix_total)

        IU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IU = IU_array.mean()

        # getConfusionMatrixPlot(confusion_matrix)
        print({'meanIU': mean_IU, 'IU_array': IU_array})
        with open(os.path.join(args.output_dir, 'result.txt'), 'w') as f:
            f.write('meanIU:' + str(mean_IU))
            f.write('\n')
            f.write('IU_array:' + str(IU_array))


    print("Finsh the preidction on whole dataset")



if __name__ == '__main__':
    main()


