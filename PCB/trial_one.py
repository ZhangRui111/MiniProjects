import imutils
import cv2

from skimage.measure import compare_ssim

imageA_path = './images/pcb_a.png'
imageB_path = './images/pcb_b.png'

# Load the two input images.
imageA = cv2.imread(imageA_path)
imageB = cv2.imread(imageB_path)

# Resize images to 400*400
imageA = imutils.resize(imageA, width=400)
imageB = imutils.resize(imageB, width=400)

# Convert the images to grayscale.
grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

# Compute the Structural Similarity Index (SSIM) between two images.
(score_ab, diff_ab) = compare_ssim(grayA, grayB, full=True)
diff_ab = (diff_ab * 255).astype("uint8")
# diff_ab = (255-diff_ab)
print("SSIM score: {}".format(score_ab))

# Threshold the difference image by cv2.THRESH_TOZERO
thresh_val, thresh_ab = cv2.threshold(
    diff_ab, 245, 255, cv2.THRESH_TOZERO)

# grayA subtract grayB
# grayA_B = grayA-grayB
# grayB_A = grayB-grayA

# Compute the Structural Similarity Index (SSIM) between grayA and grayA_B.
# (score_a_ab, diff_a_ab) = compare_ssim(grayA, grayA_B, full=True)
# diff_a_ab = (diff_a_ab * 255).astype("uint8")

# Show the output images.
# cv2.imshow("PCB_a", imageA)
# cv2.imshow("PCB_b", imageB)
cv2.imshow("Gray_a", grayA)
cv2.imshow("Gray_b", grayB)
cv2.imshow("Diff_ab", diff_ab)
cv2.imshow("Thresh_ab", thresh_ab)
# cv2.imshow("Diff_a_ab", diff_a_ab)
# cv2.imshow("GrayA_B", grayA_B)
# cv2.imshow("GrayB_A", grayB_A)

cv2.imwrite("./logs/Gray_a.png", grayA)
cv2.imwrite("./logs/Gray_b.png", grayB)
cv2.imwrite("./logs/Diff_ab.png", diff_ab)
cv2.imwrite("./logs/Thresh_ab.png", thresh_ab)

# waitKey(): makes the program wait until a key is pressed
# (at which point the script will exit).
# cv2.waitKey(0)
# cv2.destroyAllWindows()
