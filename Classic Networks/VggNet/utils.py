import skimage
import skimage.io
import skimage.transform
import numpy as np


# synset = [l.strip() for l in open('synset.txt').readlines()]


# returns image of shape [224, 224, 3]
# [height, width, depth]
def load_image(path):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0  # # Data normalization.
    assert (0 <= img).all() and (img <= 1.0).all()
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resized_img = skimage.transform.resize(crop_img, (224, 224))
    return resized_img


# returns the top1 string
def print_prob(prob, file_path):
    # # .readlines()自动将文件内容分析成一个行的列表，该列表可以由 Python 的 for... in ... 结构进行处理。
    # 另一方面，.readline()每次只读取一行，通常比 .readlines()慢得多。仅当没有足够内存可以一次读取整个文
    # 件时，才应该使用.readline()。

    # # Python strip() 方法用于移除字符串头尾指定的字符（默认为空格）。line.strip()移除每行的空格
    synset = [line.strip() for line in open(file_path).readlines()]

    # print prob
    # # np.argsort() Returns the indices that would sort an array.
    # # >>> x = np.array([3, 1, 2])
    # # >>> np.argsort(x)
    # # array([1, 2, 0])

    # # [start=start:end=end:stride=-1]  # 倒序
    # # np.argsort(prob) 根据VggNet计算出的分布概率返回的索引按照从小到大的顺序排列，所以需要倒序，
    # # 这样pred[0]就是最大的预测结果。
    pred = np.argsort(prob)[::-1]

    # Get top1 label
    top1 = synset[pred[0]]
    print(("Top1: ", top1, prob[pred[0]]))
    # Get top5 label
    top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(5)]
    print(("Top5: ", top5))
    return top1


def load_image2(path, height=None, width=None):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    if height is not None and width is not None:
        ny = height
        nx = width
    elif height is not None:
        ny = height
        nx = img.shape[1] * ny / img.shape[0]
    elif width is not None:
        nx = width
        ny = img.shape[0] * nx / img.shape[1]
    else:
        ny = img.shape[0]
        nx = img.shape[1]
    return skimage.transform.resize(img, (ny, nx))


def test():
    img = skimage.io.imread("../test_data/starry_night.jpg")
    ny = 300
    nx = img.shape[1] * ny / img.shape[0]
    img = skimage.transform.resize(img, (ny, nx))
    skimage.io.imsave("../test_data/test/output.jpg", img)


if __name__ == "__main__":
    test()
