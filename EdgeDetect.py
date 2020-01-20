import numpy as np
from scipy import ndimage
import matplotlib.pyplot as pt
import matplotlib.colors as color
import cv2 as cv


class EdgeDetect:

    def loadImage(self, imgs):
        img = cv.imread(imgs, cv.IMREAD_UNCHANGED)  # If you have a 4 channel TIFF you IMREAD_UNCHANGED or else variable will be empty

        if img.ndim > 2:
            grey_img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        return grey_img

    def loadImagePt(self, imgs):
        img = pt.imread(imgs)
        print(img.shape)
        return img

    def sobelDetect(self, img):
        #"""
        Gx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        Gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        Ix = ndimage.filters.convolve(img, Gx)
        Iy = ndimage.filters.convolve(img, Gy)

        gradient_img = np.sqrt(pow(Ix, 2.0) + pow(Iy, 2.0))
        print(gradient_img.shape)
        #"""
        """
        # Apply gray scale

        # gray_img = np.round(0.299 * img[:, :, 0] +
        # 0.587 * img[:, :, 1] +
        # 0.114 * img[:, :, 2]).astype(np.uint8)

        # Sobel Operator
        h, w = img.shape
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
                horizontalGrad = (horizontal[0, 0] * img[i - 1, j - 1]) + \
                                 (horizontal[0, 1] * img[i - 1, j]) + \
                                 (horizontal[0, 2] * img[i - 1, j + 1]) + \
                                 (horizontal[1, 0] * img[i, j - 1]) + \
                                 (horizontal[1, 1] * img[i, j]) + \
                                 (horizontal[1, 2] * img[i, j + 1]) + \
                                 (horizontal[2, 0] * img[i + 1, j - 1]) + \
                                 (horizontal[2, 1] * img[i + 1, j]) + \
                                 (horizontal[2, 2] * img[i + 1, j + 1])

                newhorizontalImage[i - 1, j - 1] = abs(horizontalGrad)

                verticalGrad = (vertical[0, 0] * img[i - 1, j - 1]) + \
                               (vertical[0, 1] * img[i - 1, j]) + \
                               (vertical[0, 2] * img[i - 1, j + 1]) + \
                               (vertical[1, 0] * img[i, j - 1]) + \
                               (vertical[1, 1] * img[i, j]) + \
                               (vertical[1, 2] * img[i, j + 1]) + \
                               (vertical[2, 0] * img[i + 1, j - 1]) + \
                               (vertical[2, 1] * img[i + 1, j]) + \
                               (vertical[2, 2] * img[i + 1, j + 1])

                newverticalImage[i - 1, j - 1] = abs(verticalGrad)

                # Edge Magnitude
                mag = np.sqrt(pow(horizontalGrad, 2.0) + pow(verticalGrad, 2.0))
                newgradientImage[i - 1, j - 1] = mag

         """
        return gradient_img


ed = EdgeDetect()
#imgs = 'C:/Users/dcampoy/Pictures/TestingCannyEdge/T0/IMG_42.tiff'
imgs = 'C:/Users/dcampoy/Pictures/Wallpapers/ar024.jpg'
# img = ed.loadImagePt(imgs)
img = ed.loadImage(imgs)
cv.imshow('Org image', img)
cv.waitKey(3000)
print(img.shape)

gradient_img = ed.sobelDetect(img)
#cv.imwrite('C:/Users/dcampoy/Pictures/TestingCannyEdge/SobelFilters/Convolve_Grad_EEG_cv(2).tiff', gradient_img)
cv.imshow('Sobel', gradient_img)
cv.waitKey(15000)
