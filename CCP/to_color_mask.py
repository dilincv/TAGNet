from PIL import Image
import numpy as np
import os

IMAGE_DIR = "/home/mingrui/CCP/CCP50/snapshots/val_output/ms_val_45000"
EXPORT_DIR = "/home/mingrui/temp"

palette = [0] * (20 * 3)
palette[0:3] = (128, 64, 128)       # 0: 'road' 
palette[3:6] = (244, 35,232)        # 1 'sidewalk'
palette[6:9] = (70, 70, 70)         # 2''building'
palette[9:12] = (102,102,156)       # 3 wall
palette[12:15] =  (190,153,153)     # 4 fence
palette[15:18] = (153,153,153)      # 5 pole
palette[18:21] = (250,170, 30)      # 6 'traffic light'
palette[21:24] = (220,220, 0)       # 7 'traffic sign'
palette[24:27] = (107,142, 35)      # 8 'vegetation'
palette[27:30] = (152,251,152)      # 9 'terrain'
palette[30:33] = ( 70,130,180)      # 10 sky
palette[33:36] = (220, 20, 60)      # 11 person
palette[36:39] = (255, 0, 0)        # 12 rider
palette[39:42] = (0, 0, 142)        # 13 car
palette[42:45] = (0, 0, 70)         # 14 truck
palette[45:48] = (0, 60,100)        # 15 bus
palette[48:51] = (0, 80,100)        # 16 train
palette[51:54] = (0, 0,230)         # 17 'motorcycle'
palette[54:57] = (119, 11, 32)      # 18 'bicycle'
palette[57:60] = (105, 105, 105)

files_name = os.listdir(IMAGE_DIR)
i = 1
for file_name in files_name:
    print(str(i) + "/" + str(len(files_name)))
    i = i + 1
    if file_name[-4:] == ".png":
        image = Image.open(os.path.join(IMAGE_DIR, file_name))
        label = np.asarray(image)
        ignore_label = 255
        id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
              3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
              7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
              14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
              18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
              28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}
        reverse=False
        label_copy = label.copy()
        if reverse:
            for v, k in id_to_trainid.items():
                label_copy[label == k] = v
        else:
            for k, v in id_to_trainid.items():
                label_copy[label == k] = v
        output_im = Image.fromarray(label_copy)
        output_im.putpalette(palette)
        output_im.save(os.path.join(EXPORT_DIR, file_name))

