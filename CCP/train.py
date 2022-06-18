import argparse
import os
import os.path as osp
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils import data
from tqdm import tqdm
from dataset.datasets import CSDataSet, CSDataSet_vis
from networks.ccpnet import Res_Deeplab
from utils.criterion import CriterionDSN, CriterionOhemDSN
from utils.encoding import DataParallelModel, DataParallelCriterion
from utils.utils import decode_labels, inv_preprocess, decode_predictions
from utils.utils import load_checkpoint
from utils.optimizer_utils import get_learning_rate, params_grouping, adjust_optimizer
from utils.utils import print_settings
from vis import vis_output
import random
torch_ver = torch.__version__[:3]

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
GROUPS = {
    # 'net.bcc.bcc.bbox_regress.fc':{'weight_decay':0},
    # 'net.bcc.bcc.bbox_regress.branch[0]':{'weight_decay':0}
}
BATCH_SIZE = 8
DATA_DIR = '/home/ubuntu/cityscapes'
DATA_LIST_ROOT = './dataset/list/cityscapes'
TRAIN_LIST_NAME = 'train.lst'
VAL_LIST_NAME = 'val.lst'
IGNORE_LABEL = 255
INPUT_SIZE = '769, 769'
LR = 1e-2
MOMENTUM = 0.9
NUM_CLASSES = 19
START_ITERS = 0
NUM_SETPS = 60000
POWER = 0.9
RANDOM_MIRROR = True
RANDOM_SCALE = True
RESTORE_FROM = None
SAVE_NUM_IMAGES = 4
SAVE_PRED_EVERY = 5000
SNAPSHOT_DIR = './debug'
WEIGHT_DECAY = 5e-4
GPU = '4,5,6,7'
OHEM = False
OHEM_THRES = 0.6
OHEM_KEEP = 200000
WARMUP_STEPS = 5000
WARMUP_START_LR = 0.0001
VIS_RAND_N_IMAGE = 7
VIS_RAND_N_POINT = 7
VIS_SPE_IMAGES = (0, 1, 2, 3, 4, 5, 6)
#VIS_SPE_POINTS = ((4, 5), (26, 23), (41, 12), (11, 27), (32, 17), (29, 48), (30, 32)) #for feature_size_with_block=49*49
VIS_SPE_POINTS = ((12, 11), (32, 19), (15, 20), (21, 29), (2, 8), (24, 31), (12, 22)) #for feature_size_with_block=33*33
LR_DECAY = 'True'
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(1234)

parser = argparse.ArgumentParser(description="CCP Network")
parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                    help="Number of images sent to the network in one step.")
parser.add_argument("--data-dir", type=str, default=DATA_DIR,
                    help="Path to the directory containing the cs dataset.")
parser.add_argument("--data-list-root", type=str, default=DATA_LIST_ROOT,
                    help="Path to the file listing the images in the dataset.")
parser.add_argument("--train-list-name", type=str, default=TRAIN_LIST_NAME,
                    help="Path to the file listing the images in the dataset.")
parser.add_argument("--val-list-name", type=str, default=VAL_LIST_NAME,
                    help="Path to the file listing the images in the dataset.")
parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                    help="The index of the label to ignore during the training.")
parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                    help="Comma-separated string with height and width of images.")
parser.add_argument("--learning-rate", type=float, default=LR,
                    help="Base learning rate for training with polynomial decay.")
parser.add_argument("--momentum", type=float, default=MOMENTUM,
                    help="Momentum component of the optimiser.")
parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                    help="Number of classes to predict (including background).")
parser.add_argument("--start-iters", type=int, default=START_ITERS,
                    help="Number of classes to predict (including background).")
parser.add_argument("--num-steps", type=int, default=NUM_SETPS,
                    help="Number of training steps.")
parser.add_argument("--power", type=float, default=POWER,
                    help="Decay parameter to compute the learning rate.")
parser.add_argument("--random-mirror", action="store_true", default=RANDOM_MIRROR,
                    help="Whether to randomly mirror the inputs during the training.")
parser.add_argument("--random-scale", action="store_true", default=RANDOM_SCALE,
                    help="Whether to randomly scale the inputs during the training.")
parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                    help="Where restore model parameters from.")
parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                    help="How many images to save.")
parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                    help="Save summaries and checkpoint every often.")
parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                    help="Where to save snapshots of the model.")
parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                    help="Regularisation parameter for L2-loss.")
parser.add_argument("--gpu", type=str, default=GPU,
                    help="choose gpu device.")
parser.add_argument("--ohem", type=bool, default=OHEM,
                    help="use hard negative mining")
parser.add_argument("--ohem-thres", type=float, default=OHEM_THRES,
                    help="choose the samples with correct probability underthe threshold.")
parser.add_argument("--ohem-keep", type=int, default=OHEM_KEEP,
                    help="choose the samples with correct probability underthe threshold.")
parser.add_argument("--warmup-steps", type=int, default=WARMUP_STEPS,
                    help="warmup-step")
parser.add_argument("--warmup-start-lr", type=float, default=WARMUP_START_LR,
                    help="warmup-start-lr")
parser.add_argument("--vis_rand-n-image", type=int, default=VIS_RAND_N_IMAGE,
                    help="")
parser.add_argument("--vis_rand-n-point", type=int, default=VIS_RAND_N_POINT,
                    help="")
parser.add_argument("--vis-spe-images", type=tuple, default=VIS_SPE_IMAGES,
                    help="")
parser.add_argument("--vis_spe_points", type=tuple, default=VIS_SPE_POINTS,
                    help="")
parser.add_argument("--lr-decay", type=str, default=LR_DECAY,
                    help="")


args = parser.parse_args()


def main():
    """Create the model and start the training."""
    # TODO reconstruct the mian func
    print_settings(args.__dict__, 'train')
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    writer = SummaryWriter(args.snapshot_dir)

    # Create network.
    cudnn.enabled = True
    deeplab = Res_Deeplab(num_classes=args.num_classes)
    if args.restore_from is not None:
        deeplab = load_checkpoint(model=deeplab, checkpoint=args.restore_from)
    model = DataParallelModel(deeplab)
    model.train()
    model.float()
    model.cuda()

    if args.ohem:
        criterion = CriterionOhemDSN(thresh=args.ohem_thres, min_kept=args.ohem_keep)
    else:
        criterion = CriterionDSN()
    criterion = DataParallelCriterion(criterion)
    criterion.cuda()

    cudnn.benchmark = True
    dataset = CSDataSet(root=args.data_dir, list_path=os.path.join(args.data_list_root, args.train_list_name), crop_size=input_size,
                        max_iters=(args.num_steps+args.warmup_steps) * args.batch_size, scale=args.random_scale,
                        mirror=args.random_mirror, mean=IMG_MEAN, ignore_label=args.ignore_label)
    dataset_vis = CSDataSet_vis(root=args.data_dir, list_path=os.path.join(args.data_list_root, args.val_list_name), max_iters=None,
                            crop_size=input_size, mean=IMG_MEAN, scale=False, mirror=False, ignore_label=args.ignore_label)
    train_loader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=5, pin_memory=True)
    params_list = params_grouping(model, GROUPS)
    optimizer = optim.SGD(params_list, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()

    pbar = tqdm(enumerate(train_loader), total=args.num_steps+args.warmup_steps, initial=args.start_iters)
    for i_iter, (images, labels, _, name) in pbar:
        i_iter += args.start_iters
        images = images.cuda()
        labels = labels.long().cuda()
        preds = model(images)
        loss = criterion(preds, labels)
        loss.backward()
        if args.lr_decay == 'True':
            lr = get_learning_rate(i_iter, args.warmup_steps, args.warmup_start_lr, args.learning_rate, args.num_steps, args.power)
        else:
            lr = args.learning_rate
        adjust_optimizer(optimizer, lr, GROUPS)

        optimizer.step()
        optimizer.zero_grad()

        if i_iter % 100 == 0:
            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], i_iter)
            writer.add_scalar('loss', loss.data.cpu().numpy(), i_iter)
        if i_iter % 500 == 0:
            vis_output(args, deeplab, dataset_vis, i_iter)
        if i_iter % 1000 == 0:
            images_inv = inv_preprocess(images, args.save_num_images, IMG_MEAN)
            labels_colors = decode_labels(labels, args.save_num_images, args.num_classes)
            preds_colors = decode_predictions(preds, args.save_num_images, args.num_classes)
            for index, (img, lab) in enumerate(zip(images_inv, labels_colors)):
                writer.add_image('Images/' + name[index], np.transpose(img[:, :, (2, 1, 0)], (2, 0, 1)), i_iter)
                writer.add_image('Labels/' + name[index], np.transpose(lab, (2, 0, 1)), i_iter)
                writer.add_image('preds/' + name[index], np.transpose(preds_colors[index], (2, 0, 1)), i_iter)

        if i_iter <= 40000:
            if i_iter % args.save_pred_every == 0:
                print('taking snapshot ...')
                torch.save(deeplab.state_dict(), osp.join(args.snapshot_dir, 'iter_' + str(i_iter) + '.pth'))
        elif i_iter >= 44000:
            if i_iter % 250 == 0:
                print('taking snapshot ...')
                torch.save(deeplab.state_dict(), osp.join(args.snapshot_dir, 'iter_' + str(i_iter) + '.pth'))
        else:
            if i_iter % 1000 == 0:
                print('taking snapshot ...')
                torch.save(deeplab.state_dict(), osp.join(args.snapshot_dir, 'iter_' + str(i_iter) + '.pth'))

        if i_iter == args.num_steps + args.warmup_steps:
            break

        pbar.set_description('[TRAIN] loss: %.4f lr: %.8f' % (loss, lr))
    writer.close()



if __name__ == '__main__':
    main()
    print('================Finish!!!================')
