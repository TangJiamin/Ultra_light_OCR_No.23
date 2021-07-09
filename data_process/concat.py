import os
import cv2
import math
import numpy as np
import random
import pandas as pd
from tqdm import tqdm
import csv

# 固定随机数种子
np.random.seed(0)

csv_file = 'feature.csv' #包含训练集图片信息的csv文件
image_path = './TrainImages' #训练集图片路径
save_path = './concat_imgs' #保存拼接后图片的路径
label_path = 'LabelTrain.txt' #训练集标签
if not os.path.exists(save_path):
    os.makedirs(save_path)
# 提取csv文件信息
message = pd.read_csv(csv_file)
image_names = message["img"]
labels = message["label"]
# label_lengths = message["label_length"]
h48_widths = message['h48_w']

# h64_widths = []
# for img_file in tqdm(os.listdir(image_path)):
#     ori_img = cv2.imread(os.path.join(image_path, img_file))
#     h = ori_img.shape[0]
#     w = ori_img.shape[1]
#     resize_h = 64
#     ratio = w / float(h)
#     resize_w = int(resize_h * ratio)
#     h64_widths.append(resize_w)
# message.insert(6, 'h64_widths', h64_widths)
# message.to_csv(csv_file, index=0)

max_str_length = 25
max_width = 512
resize_h = 48
length_threshold = 200
# 根据宽度阈值筛选小图
short_items = [[index, w] for index, w in enumerate(h48_widths) if w < length_threshold]
# single_words = [num for num, word_length in enumerate(label_lengths) if word_length == 1]

# 统一小图高度
def resize_img(ori_img, resize_h):
    h = ori_img.shape[0]
    w = ori_img.shape[1]
    ratio = w / float(h)
    resize_w = int(resize_h * ratio)
    resized_image = cv2.resize(ori_img.copy(), (resize_w, resize_h))
    # resized_image = resized_image.astype('float32')
    # if ori_img.shape[2] == 1:
    #     resized_image = resized_image / 255
    #     resized_image = resized_image[np.newaxis, :]
    # else:
    #     resized_image = resized_image.transpose((2, 0, 1)) / 255
    # resized_image -= 0.5
    # resized_image /= 0.5
    return resized_image

# 小图拼接
def concat(backgound, resized_img, label, short_items, labels, concat_num):
    # 随机选取待拼接的小图
    samples = random.sample(short_items, concat_num)
    concat_image = []
    concat_image.append(resized_img)
    concat_length = []
    concat_length.append(resized_img.shape[1])
    concat_label = []
    concat_label.append(label)
    concat_str_length = []
    concat_str_length.append(len(label))
    # 获取待拼接小图
    for sample in samples:
        index, w = sample
        # if index in single_words:
        #     con_img = cv2.imread(os.path.join(image_path, image_names[index]))
        #     # 原图的高、宽 以及通道数
        #     rows, cols, channel = con_img.shape
        #     angle = random.choice([-90, 90])
        #     # print(image_names[index] + " is rotated " + str(angle))
        #     # 绕图像的中心旋转
        #     # 参数：旋转中心 旋转度数 scale
        #     M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        #     # 参数：原始图像 旋转参数 元素图像宽高
        #     con_img = cv2.warpAffine(con_img, M, (cols, rows), borderMode=cv2.BORDER_REFLECT, borderValue=(255,255,255))
        # else:
        con_img = cv2.imread(os.path.join(image_path, image_names[index]))
        con_label = labels[index]
        # str_length = label_lengths[index]
        con_resize_img = resize_img(con_img, resize_h)
        assert con_resize_img.shape[0] == resize_h
        # cv2.imshow(image_names[index], con_resize_img)
        # cv2.waitKey(0)
        concat_w = con_resize_img.shape[1]
        concat_image.append(con_resize_img)
        concat_length.append(concat_w)
        concat_str_length.append(str_length)
        concat_label.append(con_label)
        if sum(concat_length) > max_width or sum(concat_str_length) > max_str_length:
            concat_length.pop()
            concat_image.pop()
            concat_label.pop()
            concat_str_length.pop()

    assert sum(concat_length) <= max_width, \
        "concat lenth is '{}', bigger than '{}'".format(sum(concat_length), max_width)
    assert len(concat_length) == len(concat_image) == len(concat_label)

    concat_label = "".join(concat_label)

    # new_image = np.zeros((resize_h, max_width, 3), dtype=np.uint8)
    # 在背景中随机选取插入图片的空隙
    lap = max_width - sum(concat_length)
    total_img_num = len(concat_image)
    mean_lap = int(lap / total_img_num)
    end_pos = []
    # 拼接
    if mean_lap == 0:
        stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
        _result, new_image = stitcher.stitch(concat_image)
        return new_image, concat_label
    else:
        for i in range(total_img_num):
            assert concat_length[i] == concat_image[i].shape[1], \
            "concat lenth is '{}', but img_width is '{}'".format(concat_length[i], concat_image[i].shape[1])

            if i == 0:
                inner = random.randint(0, mean_lap+1)
                backgound[:, inner:inner+concat_length[i]] = concat_image[i]
                end_pos.append(inner+concat_length[i])
            else:
                inner = random.randint(0, mean_lap+1)
                backgound[:, end_pos[i-1]+1+inner:end_pos[i-1]+1+inner+concat_length[i]] = concat_image[i]
                end_pos.append(end_pos[i-1]+1+inner+concat_length[i])
        return backgound, concat_label

# 提取训练集图片对应的标签
file_data = {}
with open(label_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        name = line.strip('\n').split('\t')[0]
        ori_label = line.strip('\n').split('\t')[-1]
        file_data[name] = ori_label

# 对每张小图做一次拼接
for item in tqdm(short_items):
    concat_num = random.randint(2, 6)
    index, w = item
    img_name = image_names[index]
    # if img_name not in file_data:
    #     print(img_name + ' is dropped')
    #     continue
    label = labels[index]
    img = cv2.imread(os.path.join(image_path, img_name))
    resized_img = resize_img(img.copy(), resize_h)
    background = cv2.imread("./bg.png")
    background = cv2.resize(background, (max_width, resize_h))
    try:
        concat_img, concat_label = concat(background, resized_img, label, short_items, labels, concat_num)
        cv2.imwrite(os.path.join(save_path, img_name), concat_img)
        file_data[img_name] = concat_label
    except:
        pass

with open('concat_train_label.txt', "w", encoding="utf-8") as f:
    for k, v in file_data.items():
        f.write(k + '\t' + v + '\n')