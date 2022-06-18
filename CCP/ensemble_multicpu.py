import os
from tqdm import tqdm
from utils.utils import get_confusion_matrix
from PIL import Image as PILImage
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from PIL import Image
import numpy as np
import multiprocessing

NPY_PATHS = ('/media/szu/mydata/wuyong/snapshots/ensemble/base/for_val/psp_ms_vote/psp_0_5',
             '/media/szu/mydata/wuyong/snapshots/ensemble/base/for_val/psp_ms_vote/psp_0_75',
             '/media/szu/mydata/wuyong/snapshots/ensemble/base/for_val/psp_ms_vote/psp_1_0',
             '/media/szu/mydata/wuyong/snapshots/ensemble/base/for_val/psp_ms_vote/psp_1_25',
             '/media/szu/mydata/wuyong/snapshots/ensemble/base/for_val/psp_ms_vote/psp_1_5',
             '/media/szu/mydata/wuyong/snapshots/ensemble/base/for_val/psp_ms_vote/psp_1_75',)

SAVE_PATH = '/media/szu/mydata/yf/danet_finetune_test'
PRED_ROOT = '/media/szu/mydata/yf/danet_finetune_test_outputs'
ENSEMBLE_TYPE = 'vote'
SOURCE = 'npy'
SAVE_NPYS = False
CAL_MIOU = True

NPY_SHAPE = (1024, 2048)
NUM_CLASS = 19
LIST_PATH = './dataset/list/cityscapes/pred.lst'
LABEL_ROOT = '/media/szu/mydata/wuyong/datasets/cityscapes'
IGNORE_LABEL = 255
N_PROCESS = 16

ID_TO_TRAINID = {-1: 255, 0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255,
                 7: 0, 8: 1, 9: 255, 10: 255, 11: 2, 12: 3, 13: 4, 14: 255, 15: 255,
                 16: 255, 17: 5, 18: 255, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
                 25: 12, 26: 13, 27: 14, 28: 15, 29: 255, 30: 255, 31: 16, 32: 17, 33: 18}


class EnsembleDataset(Dataset):
    def __init__(self, paths=NPY_PATHS, npy_shape=NPY_SHAPE, source=SOURCE, num_class=NUM_CLASS):
        self.groups_path = []
        for npy_name in os.listdir(paths[0]):
            group_path = []
            for path in paths:
                group_path.append(os.path.join(path, npy_name))
            self.groups_path.append(group_path)

        self.source = source
        self.npy_shape = npy_shape
        self.num_class = num_class

    def __getitem__(self, index):
        n_npy = []
        group_path = self.groups_path[index]
        name = group_path[0].split('/')[-1]
        for path in group_path:
            if self.source == 'npy':
                npy = np.load(path)
            elif self.source == 'pred':
                pred = Image.open(path)
                npy = np.asarray(pred)
                if sum(np.unique(npy) < self.num_class) != len(np.unique(npy)):
                    npy = convert(npy)
            assert npy.shape == self.npy_shape
            n_npy.append(npy)
        n_npy = np.stack(n_npy)
        return n_npy, name

    def __len__(self):
        return len(self.groups_path)


class Ensemble:
    def __init__(self, ensemble_type, save_path=SAVE_PATH, save_npys=SAVE_NPYS, npy_shape=NPY_SHAPE, num_class=NUM_CLASS):
        self.npy_save_path = os.path.join(save_path, 'npys')
        self.pred_save_path = os.path.join(save_path, 'outputs')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(self.pred_save_path):
            os.makedirs(self.pred_save_path)
        if not os.path.exists(self.npy_save_path):
            os.makedirs(self.npy_save_path)
        self.save_npys = save_npys
        self.ensemble_type = ensemble_type
        self.num_class = num_class
        self.noise = np.random.random_sample(npy_shape) / 1e+20

    def run(self, n_npy, name):
        if self.ensemble_type == 'avg':
            self.avg_ensemble(n_npy, name)
        elif self.ensemble_type == 'max':
            self.max_ensemble(n_npy, name)
        elif self.ensemble_type == 'custom':
            self.custom_ensemble(n_npy, name)
        elif self.ensemble_type == 'vote':
            self.vote_ensemble(n_npy, name)
        elif self.ensemble_type == 'custom_vote':
            self.custom_vote_ensemble(n_npy, name)

    def create_pred(self, npy, name):
        pred = np.asarray(np.argmax(npy, axis=0), dtype=np.uint8)
        pred = convert(pred, reverse=True)
        output_im = PILImage.fromarray(pred)
        output_im.save(os.path.join(self.pred_save_path, name + '.png'))

    def avg_ensemble(self, n_npy, name):
        ensemble_npy = n_npy.mean(axis=0)
        ensemble_npy = ensemble_npy.astype(np.float32)
        if self.save_npys:
            save_score_npy(ensemble_npy, os.path.join(self.npy_save_path, name))
        self.create_pred(ensemble_npy, name)

    def max_ensemble(self, n_npy, name):
        ensemble_npy = np.max(n_npy, axis=0)
        ensemble_npy = ensemble_npy + self.noise
        ensemble_npy = ensemble_npy.astype(np.float32)
        if self.save_npys:
            save_score_npy(ensemble_npy, os.path.join(self.npy_save_path, name))
        self.create_pred(ensemble_npy, name)

    def custom_ensemble(self, n_npy, name):
        P = [0,0,1,1,1,1,1,1,0,1,0,0,0,0,1,0,0,1,0]

        assert len(P) == self.num_class
        ensemble_npy = np.zeros(n_npy.shape[1:])
        for i in range(len(P)):
            ensemble_npy[i] = n_npy[P[i]][i]
        if self.save_npys:
            save_score_npy(ensemble_npy, os.path.join(self.npy_save_path, name))
        self.create_pred(ensemble_npy, name)

    def vote_ensemble(self, n_npy, name):
        WEIGHT = np.array([1,1,1,1,1,1], dtype=np.float64)

        assert len(WEIGHT) == len(n_npy)
        weight = WEIGHT[:, np.newaxis, np.newaxis, np.newaxis]
        vote_matrix = []
        for npy in n_npy:
            H, W = npy.shape
            predict = npy.flatten()
            one_hot = np.eye(self.num_class)[predict]
            one_hot = one_hot.reshape([H, W, one_hot.shape[-1]])
            one_hot = np.transpose(one_hot, [2, 0, 1])
            vote_matrix.append(one_hot)

        vote_matrix = np.stack(vote_matrix)
        vote_matrix = vote_matrix * weight
        vote_matrix = vote_matrix.sum(0)
        if self.save_npys:
            save_score_npy(vote_matrix, os.path.join(self.npy_save_path, name))
        self.create_pred(vote_matrix, name)

    def custom_vote_ensemble(self, n_npy, name):
        WEIGHT = np.array([1, 1], dtype=np.float64)
        P = [0,0,1,1,1,1,1,1,1,1,0,0,1,0,0,0,0,1,0]

        assert len(WEIGHT) == len(n_npy)
        assert len(P) == self.num_class
        weight = WEIGHT[:, np.newaxis, np.newaxis, np.newaxis]
        vote_matrix = []
        for npy in n_npy:
            H, W = npy.shape
            predict = npy.flatten()
            one_hot = np.eye(self.num_class)[predict]
            one_hot = one_hot.reshape([H, W, one_hot.shape[-1]])
            one_hot = np.transpose(one_hot, [2, 0, 1])
            vote_matrix.append(one_hot)

        vote_matrix = np.stack(vote_matrix)

        ensemble_npy = np.zeros(vote_matrix.shape[1:])
        for i in range(len(P)):
            ensemble_npy[i] = vote_matrix[P[i]][i]

        ensemble_npy = ensemble_npy * weight
        ensemble_npy = ensemble_npy.sum(0)
        ensemble_npy = ensemble_npy + self.noise
        ensemble_npy = np.asarray(ensemble_npy, dtype=np.float32)
        if self.save_npys:
            save_score_npy(ensemble_npy, os.path.join(self.npy_save_path, name))
        self.create_pred(ensemble_npy, name)


def save_score_npy(score, file_name):
    with open(file_name,'wb') as f:
        np.save(f, score)


def _create_predict_from_npy(path, save_path, axis, npy_name):
    npy = np.load(os.path.join(path, npy_name))
    pred = np.asarray(np.argmax(npy, axis=axis), dtype=np.uint8)
    pred = convert(pred, ID_TO_TRAINID, True)
    output_im = PILImage.fromarray(pred)
    output_im.save(os.path.join(save_path, npy_name + '.png'))


def create_predict_from_npy(path, save_path, axis):
    npy_names = os.listdir(path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    pool = multiprocessing.Pool(processes=N_PROCESS)
    for _, npy_name in enumerate(npy_names):
        pool.apply_async(_create_predict_from_npy,(path, save_path, axis, npy_name))
    print('=======creating predict from npy, please wait=======')
    pool.close()
    pool.join()
    print('=======finish=======')



def convert(x, convert_dict=ID_TO_TRAINID, reverse=False):
    x_copy = x.copy()
    if reverse:
        for v, k in convert_dict.items():
            x_copy[x == k] = v
    else:
        for k, v in convert_dict.items():
            x_copy[x == k] = v
    return x_copy


def cal_mIOU(list_path=LIST_PATH, label_root=LABEL_ROOT, pred_root=PRED_ROOT, ignore_label=IGNORE_LABEL, num_class=NUM_CLASS):

    confusion_matrix = 0
    with open(list_path, 'r') as f:
        lines = f.readlines()
    pbar = tqdm(enumerate(lines), total=len(lines))
    pbar.set_description('cal_mIOU')
    for _, line in pbar:
        pred_path = os.path.join(pred_root, line.split('\t')[0])
        label_path = os.path.join(label_root, line.split('\t')[1].strip())
        label = PILImage.open(label_path).convert("P")
        label = np.asarray(label).astype(np.int64)
        label = convert(label)
        pred = PILImage.open(pred_path).convert("P")
        pred = np.asarray(pred).astype(np.uint8)
        if sum(np.unique(pred) < num_class) != len(np.unique(pred)):
            pred = convert(pred)
        ignore_index = label != ignore_label
        label = label[ignore_index]
        pred = pred[ignore_index]
        confusion_matrix += get_confusion_matrix(label, pred, num_class)
    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)

    IU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IU = IU_array.mean()
    print({'meanIU': mean_IU, 'IU_array': IU_array})
    with open(os.path.join(os.path.dirname(pred_root), 'result.txt'), 'w') as f:
        f.write('meanIU:' + str(mean_IU))
        f.write('\n')
        f.write('IU_array:' + str(IU_array))

def main():
    ensemble_dataset = EnsembleDataset()
    ensemble_loader = DataLoader(dataset=ensemble_dataset, batch_size=N_PROCESS, shuffle=False, num_workers=N_PROCESS)
    ensemble = Ensemble(ENSEMBLE_TYPE)

    pbar = tqdm(enumerate(ensemble_loader), total=len(ensemble_loader))
    pbar.set_description('ensemble')

    # multi cpu
    for _, (n_npy, name) in pbar:
        pool = multiprocessing.Pool(processes=N_PROCESS)
        for i in range(len(n_npy)):
            pool.apply_async(ensemble.run, (n_npy[i], name[i].split('.')[0]))
        pool.close()
        pool.join()

    # single cpu for debug
    # for _, (n_npy, name) in pbar:
    #     for i in range(len(n_npy)):
    #         ensemble.run(n_npy[i], name[i].split('.')[0])


    if CAL_MIOU:
        cal_mIOU()


if __name__ == '__main__':
    #main()
    create_predict_from_npy('/media/szu/mydata/yf/ccnet_test_ohem', '/media/szu/mydata/yf/ccnet_test_ohem_outputs', 0)
    #cal_mIOU(list_path=LIST_PATH, label_root=LABEL_ROOT, pred_root=PRED_ROOT, ignore_label=IGNORE_LABEL, num_class=NUM_CLASS)
