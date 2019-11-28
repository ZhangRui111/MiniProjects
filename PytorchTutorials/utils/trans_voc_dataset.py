import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import transforms

from PIL import Image, ImageDraw, ImageFont
import numpy as np
# import sys
# import time

# sys.path.append("utils/")


def plot_bbox(img, img_ind, labels, bboxes, colors):
    """
    Plot bboxes and labels in an image.
    :param img: PIL.Image.Image
    :param img_ind: image index
    :param labels: a list consisting of all bboxes' labels.
    :param bboxes: a list consisting of all bboxes.
    :param colors: a list consisting of all bboxes' colors.
    :return:
    """
    for label, bbox, color in zip(labels, bboxes, colors):
        bbox_draw = ImageDraw.Draw(img)
        bbox_draw.rectangle(bbox, outline=color)
        label_draw = ImageDraw.Draw(img)
        label_draw.text((bbox[-2], bbox[-3]), label)
    img.show("Image with bbox")
    img.save("../logs/samples/{}".format(img_ind))


def show_sample_img(img, annota):
    """
    Show a sample image with bboxes and labels.
    :param img:
    :param annota:
    :return:
    """
    labels, bboxes, colors = [], [], []
    img_ind = annota['annotation']['filename']
    objs = annota['annotation']['object']
    n_objs = len(objs)

    if type(objs) == list:
        n_objs = len(objs)
        for i in range(n_objs):
            label = labels2val(objs[i]['name'])
            bbox = objs[i]['bndbox'].values()  # x_min, y_min, x_max, y_max
            bbox = [item for item in map(int, bbox)]
            labels.append(label)
            bboxes.append(bbox)
            colors.append('red')
    elif type(objs) == dict:
        n_objs = 1
        label = labels2val(objs['name'])
        bbox = objs['bndbox'].values()  # x_min, y_min, x_max, y_max
        bbox = [item for item in map(int, bbox)]
        labels.append(label)
        bboxes.append(bbox)
        colors.append('red')

    img.show(title="Raw image")
    plot_bbox(img, img_ind, labels, bboxes, colors)


def labels2val(label):
    trans_dict = {
        'person': 0,
        'bird': 1,
        'cat': 2,
        'cow': 3,
        'dog': 4,
        'horse': 5,
        'sheep': 6,
        'aeroplane': 7,
        'bicycle': 8,
        'boat': 9,
        'bus': 10,
        'car': 11,
        'motorbike': 12,
        'train': 13,
        'bottle': 14,
        'chair': 15,
        'diningtable': 16,
        'pottedplant': 17,
        'sofa': 18,
        'tvmonitor': 19,
    }
    return int(trans_dict[label])


def trans_voc(root, year, img_set):
    """
    Transform voc dataset into torch-style data.
    :param root:
    :param year: 2007, 2012.
    :param img_set: train, trainval, val.
    :return:
    """
    trans_dataset = {}
    dataset = torchvision.datasets.VOCDetection(root=root, year=year, image_set=img_set,
                                                transforms=None,
                                                download=True)
    # loader = DataLoader(dataset, batch_size=1, shuffle=False)
    data_size = len(dataset)
    for img, annota in dataset:
        # show_sample_img(img, annota)
        img_ind = annota['annotation']['filename']
        objs = annota['annotation']['object']
        img_w = annota['annotation']['size']['width']
        img_h = annota['annotation']['size']['height']
        img_c = annota['annotation']['size']['depth']

        if type(objs) == list:
            n_objs = len(objs)
            objs_lst = []
            for i in range(n_objs):
                label = labels2val(objs[i]['name'])
                bbox = objs[i]['bndbox'].values()  # x_min, y_min, x_max, y_max
                bbox = [item for item in map(int, bbox)]
                objs_lst.append({'label': label, 'bbox': bbox})
        elif type(objs) == dict:
            n_objs = 1
            label = labels2val(objs['name'])
            bbox = objs['bndbox'].values()  # x_min, y_min, x_max, y_max
            bbox = [item for item in map(int, bbox)]
            objs_lst = [{'label': label, 'bbox': bbox}]
        else:
            raise LookupError

        # img = np.array(img)  # convert PIL.image to numpy array in shape [H, W, C]
        # img = Image.fromarray(img)  # convert numpy array to PIL.image in size (W, H)
        # img.show()
        trans_dataset[img_ind] = {'img': img,
                                  'n_objs': n_objs,
                                  'objs': objs_lst}

    return trans_dataset


def main():
    global dev
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = trans_voc('../data/VOC/', '2007', 'train')


if __name__ == "__main__":
    main()
