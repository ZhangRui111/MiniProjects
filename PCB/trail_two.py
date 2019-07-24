import cv2
import numpy as np
import matplotlib.pyplot as plt

from geometric_trans import get_M


def img_transformation():
    """
    Crop and adopt perspective transformation on two PCBs.
    :return:
    """
    imageA_path = './images/pcb_a.png'  # [[460, 190], [1560, 187], [473, 1186], [1556, 1187]]
    imageB_path = './images/pcb_b.png'  # [[359, 145], [1639, 140], [1627, 1291], [379, 1298]]

    pcb_a = cv2.imread(imageA_path)
    pcb_b = cv2.imread(imageB_path)

    pcb_a_c = pcb_a[190:1200, 460:1600]  # [[42, 24], [1061, 18], [52, 981], [1061, 981]]
    pcb_b_c = pcb_b[140:1300, 360:1650]  # [[42, 28], [1234, 24], [61, 1139], [1224, 1135]]
    cv2.imwrite('./logs/pcb_a_c.png', pcb_a_c)
    cv2.imwrite('./logs/pcb_b_c.png', pcb_b_c)

    # x, y = img.shape[:2]
    x, y = 1300, 1300
    a2b_pers_args = [np.float32([[42, 24], [1061, 18], [52, 981], [1061, 981]]),
                     np.float32([[42, 28], [1234, 25], [63, 1139], [1224, 1136]])]
    b2b_pers_args = [np.float32([[42, 28], [1234, 25], [63, 1139], [1224, 1136]]),
                     np.float32([[42, 28], [1234, 25], [63, 1139], [1224, 1136]])]
    a2b_M = get_M("Perspective", a2b_pers_args)
    b2b_M = get_M("Perspective", b2b_pers_args)
    pcb_a2b_c = cv2.warpPerspective(pcb_a_c, a2b_M, (x, y))
    pcb_b2b_c = cv2.warpPerspective(pcb_b_c, b2b_M, (x, y))
    cv2.imshow('pcb_a2b_c', pcb_a2b_c)
    cv2.imshow('pcb_b2b_c', pcb_b2b_c)
    cv2.imwrite('./logs/pcb_a2b_c.png', pcb_a2b_c)
    cv2.imwrite('./logs/pcb_b2b_c.png', pcb_b2b_c)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def img_processing():
    # # Get gray scale images
    # imageA_path = './logs/pcb_a2b_c.png'
    # imageB_path = './logs/pcb_b2b_c.png'
    # img_a = cv2.imread(imageA_path)
    # img_b = cv2.imread(imageB_path)
    #
    # gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    # gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite('./logs/pcb_a2b_c_g.png', gray_a)
    # cv2.imwrite('./logs/pcb_b2b_c_g.png', gray_b)

    # # Get edges
    # imageA_path = './logs/pcb_a2b_c_g.png'
    # imageB_path = './logs/pcb_b2b_c_g.png'
    # pcb_a2b_c_g = cv2.imread(imageA_path)
    # pcb_b2b_c_g = cv2.imread(imageB_path)
    #
    # pcb_a2b_c_g_e = cv2.Canny(pcb_a2b_c_g, 50, 200)
    # pcb_b2b_c_g_e = cv2.Canny(pcb_b2b_c_g, 50, 200)
    # cv2.imshow('pcb_a2b_c_g_e', pcb_a2b_c_g_e)
    # cv2.imshow('pcb_b2b_c_g_e', pcb_b2b_c_g_e)
    # cv2.imwrite('./logs/pcb_a2b_c_g_e.png', pcb_a2b_c_g_e)
    # cv2.imwrite('./logs/pcb_b2b_c_g_e.png', pcb_b2b_c_g_e)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # # Re-cropping
    # imageA_path = './logs/pcb_a2b_c_g.png'
    # imageB_path = './logs/pcb_b2b_c_g.png'
    # pcb_a2b_c_g = cv2.imread(imageA_path)
    # pcb_b2b_c_g = cv2.imread(imageB_path)
    # pcb_a2b_c_g_c = pcb_a2b_c_g[400:600, 450:850]
    # pcb_b2b_c_g_c = pcb_b2b_c_g[400:600, 450:850]
    # cv2.imshow('pcb_a2b_c_g_e', pcb_a2b_c_g_c)
    # cv2.imshow('pcb_b2b_c_g_e', pcb_b2b_c_g_c)
    # cv2.imwrite('./logs/pcb_a2b_c_g_c.png', pcb_a2b_c_g_c)
    # cv2.imwrite('./logs/pcb_b2b_c_g_c.png', pcb_b2b_c_g_c)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # # Trial
    imageA_path = './logs/pcb_a2b_c_g_c.png'
    imageB_path = './logs/pcb_b2b_c_g_c.png'
    pcb_a2b_c_g_c = cv2.imread(imageA_path)
    pcb_b2b_c_g_c = cv2.imread(imageB_path)
    a_b = pcb_a2b_c_g_c-pcb_b2b_c_g_c
    thresh_val, thresh_a_b = cv2.threshold(a_b, 150, 255, cv2.THRESH_BINARY)
    # cv2.imshow('a_b', a_b)
    # cv2.imshow('thresh_a_b', a_b)
    cv2.imwrite('./logs/a_b.png', a_b)
    cv2.imwrite('./logs/thresh_a_b.png', thresh_a_b)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()




def main():
    # img_transformation()
    img_processing()


if __name__ == '__main__':
    main()
