from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2
import random
import pickle as pkl


def unique(tensor):
    """
    Get unique classes present in any given image.
    :param tensor:
    :return:
    """
    tensor_np = tensor.cpu().numpy()
    # np.unique(): Returns the sorted unique elements of an array.
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)
    # tensor_res = tensor.new_tensor(unique_tensor.shape)
    # tensor_res.copy_(unique_tensor)
    return unique_tensor


def bbox_iou(box1, box2):
    """
    Get the IoU of two bounding boxes.
    :param box1: One bounding boxes.
    :param box2: Multiple bounding boxes.
    :return:
    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # Get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # Intersection area
    # .clamp(): Clamp all elements in input into the range [min, max] and
    #           return a resulting tensor.
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
                 torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)

    # Union area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    # IoU area
    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou


def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA=True):
    """
    Take an detection feature map and turns it into a 2-D tensor, where each row of the
    tensor corresponds to attributes of a bounding box.
    Only after we have transformed our output tensors, we can concatenate the detection
    maps at three different scales (13x13, 26x26, 52x52) into one big tensor.
    :param prediction: feature map.
    :param inp_dim: dimensions of input images.
    :param anchors: dimensions of anchors.
    :param num_classes:
    :param CUDA:
    :return:
    """
    batch_size = prediction.size(0)
    grid_size = prediction.size(2)
    stride = inp_dim // grid_size
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    # .view(): reshape the tensor.
    # [1, 255, 13, 13] -> [1, 255, 169] # All shape transformations assume that grid_size=(13x13)
    prediction = prediction.view(batch_size, bbox_attrs * num_anchors, grid_size * grid_size)
    # .transpose(dim1, dim2): The given dimensions dim1 and dim2 are swapped.
    # .contiguous(): transpose() doesn't generate new tensor with new layout. The transposed
    #                tensor and original tensor are indeed sharing the memory!
    #                .contiguous() actually makes a copy of tensor as if tensor of same shape
    #                created from scratch.
    # [1, 255, 169] -> [1, 169, 255]
    prediction = prediction.transpose(1, 2).contiguous()
    # [1, 169, 255] -> [1, 507, 85]
    prediction = prediction.view(batch_size, grid_size * grid_size * num_anchors, bbox_attrs)

    # Resize anchors' dimensions from input image to feature map.
    anchors = [(a[0] / stride, a[1] / stride) for a in anchors]

    # 1. Sigmoid the centre_X, centre_Y and object confidence.
    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])

    # 2. Add the center offsets for the centre_X, centre_Y
    grid = np.arange(grid_size)
    a, b = np.meshgrid(grid, grid)
    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)
    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()
    # .repeat(): Repeats this tensor along the specified dimensions. Unlike expand(), this function
    #            copies the tensor's data to create a new tensor.
    # .unsqueeze(): Returns a new tensor with a dimension of size one inserted at the specified position.
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)

    if CUDA:
        prediction = prediction.cuda()
    prediction[:, :, :2] += x_y_offset

    # 3. Log space transform height and the width.
    anchors = torch.FloatTensor(anchors)
    if CUDA:
        anchors = anchors.cuda()
    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
    prediction[:, :, 2:4] = anchors * torch.exp(prediction[:, :, 2:4])

    # 4. Apply sigmoid activation to the the class scores
    prediction[:, :, 5: 5 + num_classes] = torch.sigmoid((prediction[:, :, 5: 5 + num_classes]))

    # 5. Resize the detections map to the size of the input image.
    prediction[:, :, :4] *= stride

    return prediction


def write_results(prediction, confidence, num_classes, nms_conf=0.4, CUDA=True):
    """
    Filter output to score threshold and apply Non-maximal suppression (NMS).
    :param prediction: [B, 10647, 85] Here, B is the number of images in a batch.
    :param confidence: score threshold
    :param num_classes: 80 for COCO
    :param nms_conf: the NMS IoU threshold
    :return: a tensor of shape D x 8, that store true detections across the entire batch.
             Here D is the true detections in all of images, each represented by a row.
             Each detections has 8 attributes, namely, index of the image in the batch to
             which the detection belongs to, 4 corner coordinates, objectness score, the
             score of class with maximum confidence, and the index of that class.
    """
    # Set the bounding box rows having a object confidence less than the threshold to zero.
    # val = (prediction[:, :, 4] > confidence)
    # val = val.float()
    conf_mask = (prediction[:, :, 4] > confidence).float().unsqueeze(2)
    prediction = prediction * conf_mask

    # Transform the (center x, center y, width, height) attributes of our boxes,
    # to (top-left corner x, top-left corner y, right-bottom corner x, right-bottom corner y).
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = (prediction[:, :, 0] - prediction[:, :, 2] / 2)
    box_corner[:, :, 1] = (prediction[:, :, 1] - prediction[:, :, 3] / 2)
    box_corner[:, :, 2] = (prediction[:, :, 0] + prediction[:, :, 2] / 2)
    box_corner[:, :, 3] = (prediction[:, :, 1] + prediction[:, :, 3] / 2)
    prediction[:, :, :4] = box_corner[:, :, :4]

    batch_size = prediction.size(0)
    write = False
    # Confidence threshold and NMS has to be done for one image at once.
    for ind in range(batch_size):
        image_pred = prediction[ind]  # image Tensor
        max_conf, max_conf_score = torch.max(image_pred[:, 5:5 + num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:, :5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)  # [10647, 7]
        # .nonzero(): Returns a tensor containing the indices of all non-zero elements of input.
        non_zero_ind = (torch.nonzero(image_pred[:, 4]))
        try:
            image_pred_ = image_pred[non_zero_ind.squeeze(), :].view(-1, 7)
        except:
            # In case of no detection in the image.
            continue

        if image_pred_.shape[0] == 0:
            continue

        # There can be multiple true detections of the same class. Get the various classes
        # detected in the image.
        img_classes = unique(image_pred_[:, -1])
        if CUDA:
            img_classes = img_classes.cuda()
        # Perform NMS
        for cls in img_classes:
            # Get the detections with one particular class.
            cls_mask = image_pred_ * (image_pred_[:, -1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:, -2]).squeeze()
            image_pred_class = image_pred_[class_mask_ind].view(-1, 7)

            # Sort detections such that the entry with the maximum confidence is at the top.
            conf_sort_index = torch.sort(image_pred_class[:, 4], descending=True)[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)  # Number of detections

            # Eliminate duplicate detections.
            for i in range(idx):
                # Get the IOUs of all boxes that come after the one we are looking at.
                try:
                    ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i + 1:])
                # At that point (ValueError or IndexError), we can ascertain that NMS can remove no
                # further bounding boxes, and we break out of the loop.
                except ValueError:
                    # As we proceed with the loop, a number of bounding boxes may be removed from
                    # image_pred_class.
                    # In case of the slice image_pred_class[i+1:] may return an empty tensor.
                    break
                except IndexError:
                    # As we proceed with the loop, a number of bounding boxes may be removed from
                    # image_pred_class.
                    # This means, we cannot have idx iterations.
                    break

                # Zero out all the detections that have IoU > threshold with the target detection
                # that has the maximum confidence.
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_class[i + 1:] *= iou_mask

                # Remove the zero entries (duplicated detections) from predictions.
                non_zero_ind = torch.nonzero(image_pred_class[:, 4]).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1, 7)

            # Repeat the batch_id for as many detections of the class cls in the image
            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)
            seq = batch_ind, image_pred_class

            if not write:
                output = torch.cat(seq, 1)
                write = True
            else:
                out = torch.cat(seq, 1)
                output = torch.cat((output, out))

    try:
        return output
    except:
        #  Return 0 if there's hasn't been a single detection in any images of the batch.
        return 0


def letterbox_image(img, inp_dim):
    """
    Resize the image, while keeping the aspect ratio consistent, and padding the left
    out areas with (128,128,128)
    :param img:
    :param inp_dim:
    :return:
    """
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w / img_w, h / img_h))
    new_h = int(img_h * min(w / img_w, h / img_h))
    resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)
    canvas[(h - new_h) // 2:(h - new_h) // 2 + new_h, (w - new_w) // 2:(w - new_w) // 2 + new_w, :] = resized_image
    return canvas


def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 
    OpenCV loads an image as an numpy array, with BGR as the order of the color channels.
    PyTorch's image input format is (Batches x Channels x Height x Width), ordered in RGB.
    """
    img = (letterbox_image(img, (inp_dim, inp_dim)))
    img = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img


def load_classes(namesfile):
    """
    Returns a dictionary which maps the index of every class to a string of it's name.
    :param namesfile:
    :return:
    """
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names


def plot_bbx(x, results):
    """
    Plot detected bounding boxes in the image.
    :param x:
    :param results:
    :return:
    """
    classes = load_classes("data/coco.names")
    # "pallete" is a pickled file that contains many colors to randomly choose from.
    colors = pkl.load(open("data/pallete", "rb"))

    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = results[int(x[0])]
    cls = int(x[-1])
    color = random.choice(colors)
    label = "{0}".format(classes[cls])
    cv2.rectangle(img, c1, c2, color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    # -1 argument of the cv2.rectangle function is used for creating a filled rectangle
    cv2.rectangle(img, c1, c2, color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1);
    return img
