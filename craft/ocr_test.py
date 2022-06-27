import cv2
import easyocr
import numpy as np

import os
os.environ["KMP_DUPLICATE_LIB_OK"]='True'

img = cv2.imread('outputs1/casio_crop_normal.jpg')
kernel = np.ones((5,5), np.uint8)
# Setting parameter values
t_lower = 50  # Lower Threshold
t_upper = 150  # Upper threshold

img_erosion = cv2.erode(img,kernel, iterations = 2)
# Applying the Canny Edge filter
edge = cv2.Canny(img_erosion, t_lower, t_upper)

cv2.imwrite('outputs1/edge.png', edge)
cv2.imwrite('outputs1/img_erosin.png', img_erosion)

reader = easyocr.Reader(['en']) # this needs to run 33only once to load the model into memory
result = reader.readtext(img_erosion)

print(result)

cv2.waitKey(0)
cv2.destroyAllWindows()