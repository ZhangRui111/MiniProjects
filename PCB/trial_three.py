import cv2
import numpy as np


def show_write_imgs(img_a, img_b, save_path_a=None, save_path_b=None):
    """
    Show two images [and save these images] by OpenCV.
    :param img_a:
    :param img_b:
    :param save_path_a:
    :param save_path_b:
    :return:
    """
    cv2.imshow('pcb_a', img_a)
    cv2.imshow('pcb_b', img_b)
    if save_path_a is not None:
        cv2.imwrite(save_path_a, img_a)
    if save_path_b is not None:
        cv2.imwrite(save_path_b, img_b)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def compute_diff(kp_a, des_a, kp_b, des_b):
    if len(kp_a) != 0 and len(kp_b) != 0:
        # # Create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        # # Match descriptors.
        matches = bf.match(des_a, des_b)
        # # Sort them in the order of their distance.
        matches = sorted(matches, key=lambda x: x.distance)
        # # Get an average keypoint distance.
        len_matches = len(matches)
        sum_distances = 0
        for i in range(len_matches):
            sum_distances += matches[i].distance
        return sum_distances/len_matches
    else:
        return 0


def draw_diff_arr(img_a, img_b, diff_arr, x, y):
    # # Normalize image to between 0 and 255
    diff_arr *= (255.0 / diff_arr.max())
    threshold_per = 0.6
    threshold = np.sort(diff_arr.ravel())[int(diff_arr.ravel().shape[0] * threshold_per)]

    img = np.zeros((x, y), np.uint8)  # (x, y) for grayscale image.
    # img = np.zeros((x, y, 3), np.uint8)  # (x, y, 3) for colored image.
    rows = diff_arr.shape[0]
    cols = diff_arr.shape[1]
    for i in range(rows):
        for j in range(cols):
            print("patch's color:{0}".format(int(diff_arr[i][j])))
            cv2.rectangle(img, (j*100, i*100), ((j+1)*100, (i+1)*100), int(diff_arr[i][j]),
                          thickness=cv2.FILLED)
            if diff_arr[i][j] > threshold:
                cv2.rectangle(img_a, (j * 100, i * 100), ((j + 1) * 100, (i + 1) * 100), (0, 0, 255), 4)
                cv2.rectangle(img_b, (j * 100, i * 100), ((j + 1) * 100, (i + 1) * 100), (0, 0, 255), 4)
            # cv2.imshow('img', img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
    # cv2.imshow('img', img)
    cv2.imwrite("./logs/trial_three/diff_result.png".format(threshold_per), img)
    cv2.imwrite("./logs/trial_three/{}_img_a_diff.png".format(threshold_per), img_a)
    cv2.imwrite("./logs/trial_three/{}_img_b_diff.png".format(threshold_per), img_b)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def main():
    imageA_path = './logs/trial_two/pcb_a2b_c.png'
    imageB_path = './logs/trial_two/pcb_b2b_c.png'
    pcb_a = cv2.imread(imageA_path)
    pcb_b = cv2.imread(imageB_path)
    pcb_a_g = cv2.cvtColor(pcb_a, cv2.COLOR_BGR2GRAY)
    pcb_b_g = cv2.cvtColor(pcb_b, cv2.COLOR_BGR2GRAY)

    sha_pcb_a, sha_pcb_b = pcb_a_g.shape, pcb_b_g.shape
    assert sha_pcb_a == sha_pcb_b
    print("pcb_a's shape: {0}\npcb_b's shape: {1}".format(sha_pcb_a, sha_pcb_b))
    # # Divide image into rows * cols patches.
    row_stride, col_stride = 50, 50  # stride along the vertical and horizontal direction.
    rows = int(sha_pcb_a[0] / row_stride)
    cols = int(sha_pcb_a[1] / col_stride)

    # # Construct a SIFT object with optional different thresholds.
    # sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=100, edgeThreshold=0)
    # sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.01)
    sift = cv2.xfeatures2d.SIFT_create()
    # # Construct a SUFT object.
    # suft = cv2.xfeatures2d.SUFT_create()
    # # Construct a ORB object.
    # orb = cv2.ORB_create(nfeatures=1500)
    diff_arr = np.zeros((rows, cols))  # save sift distance for every patch.
    min_diff = 1000

    counter = 0
    zero_counter = 0
    for i in range(rows):
        for j in range(cols):
            counter += 1
            patch_a = pcb_a_g[i * row_stride:(i + 1) * row_stride, j * col_stride:(j + 1) * col_stride]
            patch_b = pcb_b_g[i * row_stride:(i + 1) * row_stride, j * col_stride:(j + 1) * col_stride]
            # show_write_imgs(patch_a, patch_b)
            kp_a, des_a = sift.detectAndCompute(patch_a, None)
            kp_b, des_b = sift.detectAndCompute(patch_b, None)
            # # Draw a circle with size of keypoint and show its orientation.
            # patch_a_d = cv2.drawKeypoints(patch_a, kp_a, outImage=1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            # patch_b_d = cv2.drawKeypoints(patch_b, kp_b, outImage=1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            # show_write_imgs(patch_a_d, patch_b_d)

            diff_val = compute_diff(kp_a, des_a, kp_b, des_b)
            if diff_val == 0:
                zero_counter += 1
            diff_arr[i][j] = diff_val
            if diff_val < min_diff and diff_val != 0:
                min_diff = diff_val
            print("Number of keypoints -- patch_a: {0} | patch_b: {1}".format(len(kp_a), len(kp_b)))

    print(diff_arr)
    diff_arr_nonzero = diff_arr.copy()
    inds = np.where(diff_arr_nonzero < 1)
    for k in range(len(inds[0])):
        diff_arr_nonzero[inds[0][k]][inds[1][k]] = min_diff
    draw_diff_arr(pcb_a, pcb_b, diff_arr_nonzero, sha_pcb_a[0], sha_pcb_a[1])
    print("zero keypoints: {}".format(zero_counter/counter))
    # show_write_imgs(pcb_a_g, pcb_b_g, './logs/trial_three/', './logs/trial_three/')
    # show_write_imgs(pcb_a_g, pcb_b_g)


if __name__ == '__main__':
    main()
