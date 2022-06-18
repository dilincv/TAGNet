import os
import math
import random
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
from torch.nn.parallel.data_parallel import DataParallel
from torch.nn.parallel.scatter_gather import scatter
from PIL import Image, ImageOps, ImageFilter
import torch.utils.data as data
def test_batchify_fn(data):
    x = data[0]
    error_msg = "batch must contain tensors, tuples or lists; found {}"
    if isinstance(data[0], (str, torch.Tensor)):
        return list(data)

    elif isinstance(data[0], (tuple, list)):
        data = zip(*data)
        return [test_batchify_fn(i) for i in data]
    raise TypeError((error_msg.format(type(data[0]))))


class BaseDataset(data.Dataset):
    def __init__(self, root, split, mode=None, transform=None,
                 target_transform=None, base_size=520, crop_size=480, ignore_label=255):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.split = split
        self.mode = mode if mode is not None else split
        self.base_size = base_size
        self.crop_size = crop_size
        self.ignore_label = ignore_label
        if self.mode == 'train':
            print('BaseDataset: base_size {}, crop_size {}'. \
                format(base_size, crop_size))

    def __getitem__(self, index):
        raise NotImplemented

    @property
    def num_class(self):
        return self.NUM_CLASS

    @property
    def pred_offset(self):
        raise NotImplemented

    def _val_sync_transform(self, img, mask):
        outsize = self.crop_size
        short_size = outsize
        w, h = img.size
        if w > h:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - outsize) / 2.))
        y1 = int(round((h - outsize) / 2.))
        img = img.crop((x1, y1, x1+outsize, y1+outsize))
        mask = mask.crop((x1, y1, x1+outsize, y1+outsize))
        img.show()
        # final transform
        return img, self._mask_transform(mask)

    def _sync_transform(self, img, mask):
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        # random scale (short edge from 480 to 720)
        short_size = random.randint(int(self.base_size*0.5), int(self.base_size*2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # random rotate -10~10, mask using NN rotate
        deg = random.uniform(-10, 10)
        img = img.rotate(deg, resample=Image.BILINEAR)
        mask = mask.rotate(deg, resample=Image.NEAREST)
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.ignore_label)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1+crop_size, y1+crop_size))
        mask = mask.crop((x1, y1, x1+crop_size, y1+crop_size))
        # gaussian blur as in PSP
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
        # final transform
        return img, self._mask_transform(mask)

    def _mask_transform(self, mask):
        return torch.from_numpy(np.array(mask)).long()


class CoCoStuff(BaseDataset):

    def __init__(self, root="/home/siting/Yuanfeng/dataset/coco", split='train2017', max_iters=None,
                 mode=None, transform=None, target_transform=None, return_img_id = False, **kwargs):
        super(CoCoStuff, self).__init__(
            root, split, mode, transform, target_transform, **kwargs)

        list_file = os.path.join(self.root, "annotations", "imagelist", split+".txt")

        self.img_ids = [line.strip().split() for line in open(list_file)]

        self.imgDir = os.path.join(self.root, split)

        self.maskDir = os.path.join(self.root, "annotations", split)

        self.transform = transform

        self.target_transform = target_transform

        self.return_img_id = return_img_id

        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
            pass

    def __getitem__(self, index):
        img_id = self.img_ids[index][0]

        img = Image.open(os.path.join(self.imgDir, img_id + '.jpg')).convert('RGB')
        # img.show()
        if self.mode == 'test2017':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(img_id)

        mask = Image.open(os.path.join(self.maskDir, img_id + ".png")).convert('P')
        # print(np.unique(mask))
        #TODO implement a fuc to inoput a mask and return a PIL (mask+color), could be directy show

        if self.mode == 'train2017':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val2017':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            mask = self._mask_transform(mask)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)
        if self.return_img_id:
            return img, mask, img_id
        else:
            return img, mask

    def _mask_transform(self, mask):
        target = np.array(mask).astype('int32')
        return torch.from_numpy(target).long()

    def __len__(self):
        return len(self.img_ids)

    @property
    def pred_offset(self):
        return 1


class MultiEvalModule(DataParallel):
    """Multi-size Segmentation Eavluator"""
    def __init__(self, module, nclass, device_ids=None, flip=True,scales=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
                 base_size=2048, crop_size=769):

        super(MultiEvalModule, self).__init__(module, device_ids)
        self.nclass = nclass
        self.base_size = base_size
        self.crop_size = crop_size
        self.scales = scales
        self.flip = flip
        self.mean = [.0, .0, .0]
        self.std = [1, 1, 1]
        print('scales: {}'.format(self.scales))
        print('MultiEvalModule: base_size {}, crop_size {}'. \
            format(self.base_size, self.crop_size))

    def parallel_forward(self, inputs, **kwargs):
        """Multi-GPU Mult-size Evaluation

        Args:
            inputs: list of Tensors
        """
        inputs = [(input.unsqueeze(0).cuda(device),)
                  for input, device in zip(inputs, self.device_ids)]
        replicas = self.replicate(self, self.device_ids[:len(inputs)])
        kwargs = scatter(kwargs, target_gpus, dim) if kwargs else []
        if len(inputs) < len(kwargs):
            inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
        elif len(kwargs) < len(inputs):
            kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
        outputs = self.parallel_apply(replicas, inputs, kwargs)
        return outputs

    def forward(self, image):
        """Mult-size Evaluation"""
        # only single image is supported for evaluation
        batch, _, h, w = image.size()
        assert(batch == 1)
        stride_rate = 2.0/3.0
        crop_size = self.crop_size
        stride = int(crop_size * stride_rate)
        with torch.cuda.device_of(image):
            scores = image.new().resize_(batch,self.nclass,h,w).zero_().cuda()

        for scale in self.scales:
            long_size = int(math.ceil(self.base_size * scale))
            if h > w:
                height = long_size
                width = int(1.0 * w * long_size / h + 0.5)
                short_size = width
            else:
                width = long_size
                height = int(1.0 * h * long_size / w + 0.5)
                short_size = height
            # resize image to current size
            cur_img = F.upsample(image, size=(height, width), mode='bilinear', align_corners=True)

            if long_size <= crop_size:
                pad_img = pad_image(cur_img, self.mean,
                                    self.std, crop_size)
                outputs = module_inference(self.module, pad_img, self.flip)
                outputs = crop_image(outputs, 0, height, 0, width)

            else:
                if short_size < crop_size:
                    # pad if needed
                    pad_img = pad_image(cur_img, self.mean,
                                        self.std, crop_size)
                else:
                    pad_img = cur_img
                _,_,ph,pw = pad_img.size()
                assert(ph >= height and pw >= width)
                # grid forward and normalize
                h_grids = int(math.ceil(1.0 * (ph-crop_size)/stride)) + 1
                w_grids = int(math.ceil(1.0 * (pw-crop_size)/stride)) + 1
                with torch.cuda.device_of(image):
                    outputs = image.new().resize_(batch,self.nclass, ph, pw).zero_().cuda()
                    count_norm = image.new().resize_(batch, 1, ph, pw).zero_().cuda()
                # grid evaluation
                for idh in range(h_grids):
                    for idw in range(w_grids):
                        h0 = idh * stride
                        w0 = idw * stride
                        h1 = min(h0 + crop_size, ph)
                        w1 = min(w0 + crop_size, pw)
                        crop_img = crop_image(pad_img, h0, h1, w0, w1)
                        # pad if needed
                        pad_crop_img = pad_image(crop_img, self.mean,
                                                 self.std, crop_size)
                        output = module_inference(self.module, pad_crop_img, self.flip)
                        outputs[:, :, h0:h1, w0:w1] += crop_image(output, 0, h1-h0, 0, w1-w0)
                        count_norm[:, :, h0:h1, w0:w1] += 1
                assert((count_norm==0).sum()==0)
                outputs = outputs / count_norm
                outputs = outputs[:, :, :height, :width]

            score = F.upsample(outputs, size=(h, w), mode='bilinear', align_corners=True)
            scores += score

        return scores


def module_inference(module, image, flip=True):
    #input:[1, 3, crop_height, crop_width]
    #attention: this network has two output, so we choose [0]

    _, _, crop_height, crop_width = image.size()
    output = module(image)
    if isinstance(output, (tuple, list)):
        output = output[0]

    # reszie the score to the cropsize
    output = F.upsample(input=output, size=(crop_height, crop_width), mode="bilinear", align_corners=True)

    if flip:
        fimg = flip_image(image)
        foutput = module(fimg)
        if isinstance(foutput, (tuple, list)):
            foutput = foutput[0]
        foutput = F.upsample(input=foutput, size=(crop_height, crop_width),
                            mode="bilinear", align_corners=True)
        output += flip_image(foutput)
    return output.exp()

def resize_image(img, h, w, **up_kwargs):
    return F.upsample(img, (h, w), **up_kwargs)

def pad_image(img, mean, std, crop_size):
    b,c,h,w = img.size()
    assert(c==3)
    padh = crop_size - h if h < crop_size else 0
    padw = crop_size - w if w < crop_size else 0
    pad_values = -np.array(mean) / np.array(std)
    img_pad = img.new().resize_(b,c,h+padh,w+padw)
    for i in range(c):
        # note that pytorch pad params is in reversed orders
        img_pad[:,i,:,:] = F.pad(img[:,i,:,:], (0, padw, 0, padh), value=pad_values[i])
    assert(img_pad.size(2)>=crop_size and img_pad.size(3)>=crop_size)
    return img_pad

def crop_image(img, h0, h1, w0, w1):
    return img[:,:,h0:h1,w0:w1]

def flip_image(img):
    assert(img.dim()==4)
    with torch.cuda.device_of(img):
        idx = torch.arange(img.size(3)-1, -1, -1).type_as(img).long()
    return img.index_select(3, idx)

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)

    hist = np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask],
        minlength=n_class ** 2,
    ).reshape(n_class, n_class)
    return hist

if __name__ == '__main__':

    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([.485, .456, .406], [.229, .224, .225])])

    Bonedataset = CoCoStuff(root="/home/siting/Yuanfeng/dataset/coco", split="train2017", mode='train2017', transform=train_transform,max_iters=60000,
                            target_transform=None, base_size=513, crop_size=321, return_img_id=True)

    loader = data.DataLoader(
        Bonedataset,
        batch_size=1,
        num_workers=1,
        shuffle=False,
        collate_fn=test_batchify_fn

    )

    mean = 0.
    std = 0.
    nb_samples = 0.
    for i, (image, labels, filenames) in enumerate(loader):
        print(np.unique(labels[0].numpy()))
        pass

    mean /= nb_samples
    std /= nb_samples

    print(mean)
    print(std)
