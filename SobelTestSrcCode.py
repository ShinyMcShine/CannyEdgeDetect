import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2 as cv

# Open the image
img = np.array(Image.open('C:/Users/dcampoy/Pictures/TestingCannyEdge/T1/IMG_42.tiff'))
#img = np.array(Image.open('C:/Users/dcampoy/Pictures/Wallpapers/ar024.jpg'))

# Apply gray scale if RGB or any other 3 channel image
if img.ndim > 2:
    gray_img = np.round(0.299 * img[:, :, 0] +
                    0.587 * img[:, :, 1] +
                    0.114 * img[:, :, 2]).astype(np.uint8)
else:
    gray_img = img

# Sobel Operator
h, w = gray_img.shape
# define filters
horizontal = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # s2
vertical = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  # s1

# define images with 0s
newhorizontalImage = np.zeros((h, w))
newverticalImage = np.zeros((h, w))
newgradientImage = np.zeros((h, w))

# offset by 1
for i in range(1, h - 1):
    for j in range(1, w - 1):
        horizontalGrad = (horizontal[0, 0] * gray_img[i - 1, j - 1]) + \
                         (horizontal[0, 1] * gray_img[i - 1, j]) + \
                         (horizontal[0, 2] * gray_img[i - 1, j + 1]) + \
                         (horizontal[1, 0] * gray_img[i, j - 1]) + \
                         (horizontal[1, 1] * gray_img[i, j]) + \
                         (horizontal[1, 2] * gray_img[i, j + 1]) + \
                         (horizontal[2, 0] * gray_img[i + 1, j - 1]) + \
                         (horizontal[2, 1] * gray_img[i + 1, j]) + \
                         (horizontal[2, 2] * gray_img[i + 1, j + 1])

        newhorizontalImage[i - 1, j - 1] = abs(horizontalGrad)

        verticalGrad = (vertical[0, 0] * gray_img[i - 1, j - 1]) + \
                       (vertical[0, 1] * gray_img[i - 1, j]) + \
                       (vertical[0, 2] * gray_img[i - 1, j + 1]) + \
                       (vertical[1, 0] * gray_img[i, j - 1]) + \
                       (vertical[1, 1] * gray_img[i, j]) + \
                       (vertical[1, 2] * gray_img[i, j + 1]) + \
                       (vertical[2, 0] * gray_img[i + 1, j - 1]) + \
                       (vertical[2, 1] * gray_img[i + 1, j]) + \
                       (vertical[2, 2] * gray_img[i + 1, j + 1])

        newverticalImage[i - 1, j - 1] = abs(verticalGrad)

        # Edge Magnitude
        mag = np.sqrt(pow(horizontalGrad, 2.0) + pow(verticalGrad, 2.0))
        newgradientImage[i - 1, j - 1] = mag
print(newgradientImage.shape)
#plt.figure()
#plt.title('EEG')
#cv.imwrite('C:/Users/dcampoy/Pictures/TestingCannyEdge/SobelFilters/ForLoop_Grad_EEG(3).tiff', newgradientImage)
#cv.imshow('EEG', newgradientImage)
#cv.waitKey(5000)
#plt.imsave('C:/Users/dcampoy/Pictures/TestingCannyEdge/SobelFilters/ForLoop_Grad_EEG_plt(3).tiff', newgradientImage, format='tiff', cmap='gray')
plt.imshow(newgradientImage, cmap='gray')
plt.show()