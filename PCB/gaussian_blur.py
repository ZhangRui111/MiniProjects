import cv2
import numpy as np
import skimage.measure

# # Read image and convert it to gray scale
src = cv2.imread('./images/num_13.jpg')
src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
# # Gaussian blur
dst = cv2.GaussianBlur(src, (5, 5), cv2.BORDER_DEFAULT)
# # Down-sampling
dwp = skimage.measure.block_reduce(dst, (5, 5), np.max)
cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.imshow('dwp', dwp)
cv2.imwrite('./logs/blur/src.png', src)
cv2.imwrite('./logs/blur/dst.png', dst)
cv2.imwrite('./logs/blur/dwp.png', dwp)
cv2.waitKey(0)
cv2.destroyAllWindows()
