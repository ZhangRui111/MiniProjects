import imutils
import cv2

from skimage.measure import compare_ssim

imageA_path = './images/icon.jpg'
imageB_path = './images/icon_copy.jpg'

# Load the two input images.
imageA = cv2.imread(imageA_path)
imageB = cv2.imread(imageB_path)

# Convert the images to grayscale.
grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

# Compute the Structural Similarity Index (SSIM) between the two
# images, ensuring that the difference image is returned.
(score, diff) = compare_ssim(grayA, grayB, full=True)
diff = (diff * 255).astype("uint8")
print("SSIM: {}".format(score))

# Threshold the difference image, followed by finding contours to
# obtain the regions of the two input images that differ.
thresh_val, thresh = cv2.threshold(
    diff, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
cnts = cv2.findContours(
    thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# Loop over the contours.
for c in cnts:
    # Compute the bounding box of the contour and then draw the
    # bounding box on both input images to represent where the
    # two images differ.
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)

# Show the output images.
cv2.imshow("Original", imageA)
cv2.imshow("Modified", imageB)
cv2.imshow("Diff", diff)
cv2.imshow("Thresh", thresh)
# waitKey(): makes the program wait until a key is pressed
# (at which point the script will exit).
cv2.waitKey(0)
cv2.destroyAllWindows()
