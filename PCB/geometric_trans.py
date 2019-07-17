"""
Different geometric transformation to images like
translation, rotation, affine transformation etc.
"""
import cv2
import numpy as np


def get_M(trans_type, args_list):
    """
    Get the Transformation Matrix M.
    Support four Transformation: translation, rotation, affine, perspective, shearing.
    :return: M
    """
    if trans_type == "Translation":
        # # args_list[0] -- t_x
        # # args_list[1] -- t_y
        assert len(args_list) == 2
        return np.float32([[1, 0, args_list[0]], [0, 1, args_list[1]]])
    elif trans_type == "Rotation":
        # # args_list[0] -- rotation center
        # # args_list[1] -- rotation angle
        # # args_list[2] -- scale factor
        assert len(args_list) == 3
        return cv2.getRotationMatrix2D(args_list[0], args_list[1], args_list[2])
    elif trans_type == "Affine":
        # # args_list[0] -- 3 points before affine, shape (3, 2), dtype float.
        # # args_list[1] -- 3 points after affine, shape (3, 2), dtype float.
        assert len(args_list) == 2
        return cv2.getAffineTransform(args_list[0], args_list[1])
    elif trans_type == "Perspective":
        # # args_list[0] -- 4 points before affine, shape (4, 2), dtype float.
        # # args_list[1] -- 4 points after affine, shape (4, 2), dtype float.
        assert len(args_list) == 2
        return cv2.getPerspectiveTransform(args_list[0], args_list[1])
    elif trans_type == "Shearing_horizontal":
        # # args_list[0] -- shearing factor in horizontal direction.
        assert len(args_list) == 1
        return np.float32([[1, args_list[0], 0], [0, 1, 0]])
    elif trans_type == "Shearing_vertical":
        # # args_list[0] -- shearing factor in vertical direction.
        assert len(args_list) == 1
        return np.float32([[1, 0, 0], [args_list[0], 1, 0]])
    else:
        raise Exception("{} is not a valid transformation.".format(type))


def my_resize(img, interpolation=cv2.INTER_LINEAR, d_size=None, fx=None, fy=None):
    """
    :param img: source/input image.
    :param interpolation: flag that takes one of the following methods.
        INTER_NEAREST – a nearest-neighbor interpolation. Recommended when shrinking.
        INTER_LINEAR – a bilinear interpolation (used by default). Recommended when enlarging.
        INTER_AREA – resampling using pixel area relation. Recommended when enlarging.
    :param d_size: desired size for the output image.
    :param fx: scale factor along the horizontal axis.
    :param fy: scale factor along the vertical axis.
    :return:
    """
    if d_size is not None:
        res = cv2.resize(img, d_size, interpolation=interpolation)
    elif d_size is None:
        res = cv2.resize(img, None, fx=fx, fy=fy, interpolation=interpolation)
    else:
        raise ValueError("resize error")
    cv2.imshow('img', img)
    cv2.imshow('res', res)
    # cv2.imwrite('./logs/img_.jpg', img)
    # cv2.imwrite('./logs/res_.jpg', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    img = cv2.imread('./images/img_affine_large.jpg')
    # # ------------------ Scaling ------------------- # #
    # rows, cols = img.shape[:2]
    # my_resize(img, cv2.INTER_CUBIC, d_size=(2 * rows, 2 * cols))

    # # --------------- Transformation --------------- # #
    # # Translation, Rotation, Affine, Perspective, Shearing.
    # # cv2.warpAffine(image, Transformation Matrix, size of image after Transformation)
    rows, cols = img.shape[:2]
    # rotation_args = [(rows/2, cols/2), 45, 1]
    # affine_args = [np.float32([[122, 50], [58, 92], [153, 92]]),
    #                np.float32([[97, 76], [44, 140], [166, 107]])]
    reverse_affine_args = [np.float32([[97, 76], [44, 140], [166, 107]]),
                           np.float32([[122, 50], [58, 92], [153, 92]])]
    # perspective_args = [np.float32([[56, 65], [238, 52], [28, 237], [239, 240]]),
    #                     np.float32([[0, 0], [200, 0], [0, 200], [200, 200]])]
    M = get_M("Affine", reverse_affine_args)
    # res = cv2.warpAffine(img, M, (rows * 2, cols * 2))
    # res = cv2.warpAffine(img, M, (rows, cols))
    res = cv2.warpAffine(img, M, (int(rows/2), int(cols/2)))
    # res = cv2.warpPerspective(img, H, (rows, cols))
    cv2.imshow('img', img)
    cv2.imshow('res', res)
    # cv2.imwrite('./logs/img_.jpg', img)
    cv2.imwrite('./images/img_affine_large_recover.jpg', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
