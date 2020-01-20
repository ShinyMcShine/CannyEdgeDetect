from scipy import ndimage
from scipy.ndimage.filters import convolve
from scipy import misc
import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


### Code below is based on https://github.com/FienSoP/canny_edge_detector.git ###
class CannyEdge:
    def __int__(self, imgs, sigma=1, krnl_size=5, weak_pixel=75, strg_pixel=255, low_thres=0.05, high_thresh=0.15):
        self.imgs = imgs
        self.highlight_imgs = []
        self.img_smooth = None
        self.gradMat = None
        self.thetaMat = None
        self.nonMaxImg = None
        self.threshImg = None
        self.weak_pixel = weak_pixel
        self.strg_pixel = strg_pixel
        self.sigma = sigma
        self.krnl_size = krnl_size
        self.low_thresh = low_thres
        self.high_thresh = high_thresh
        return

    def gryimage(self, img):
        print('Converting Image to Grayscale.....\n')
        self.imgs = plt.imread(img)
        print('Img Dims   \n', self.imgs.shape)
        return self.imgs

    def dspimg(self, img):
        plt.imshow('Image', img)

    def applyBlur(self, img):  # To visualize the blur for a single img
        self.img_smooth = convolve(img, self.gausKrnl(self.krnl_size, self.sigma))
        # self.dspimg(self.img_smooth)   # to display image for testing
        return self.img_smooth

    def gausKrnl(self, size, sigma=1):
        size = int(size) // 2  # take the floor result
        x, y = np.mgrid[-size:size + 1, -size:size + 1]  # creates a grid of coordinates
        # Kernel size and values #
        '''
        print ('kernel size   \n', size)
        print('x result    \n', x)
        print('y result    \n', y)
        '''
        normal = 1 / (2.0 * np.pi * sigma ** 2)
        g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma))) * normal
        return g

    def sobelFilter(self, img):
        # Calculate Gradient to for edge detection using Sobel filter
        Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)  # Sobel kernel horizontal filter
        Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)  # Sobel kernel vertical filter

        Ix = ndimage.filters.convolve(img, Kx)  # Calculate the derivative using Kx
        Iy = ndimage.filters.convolve(img, Ky)  # Calculate the derivative using Ky

        G = np.hypot(Ix, Iy)  # Calculate magnitude of gradient
        G = G / G.max() * 255
        theta = np.arctan2(Iy, Ix)  # gradient slope
        return G, theta

    def nonMaxSupress(self, img, D):
        M, N = img.shape
        Z = np.zeros((M, N), dtype=np.int32)
        angle = D * 180. / np.pi
        angle[angle < 0] += 180

        for i in range(1, M - 1):
            for j in range(1, N - 1):
                try:
                    q = 255
                    r = 255

                    # angle measure of 0
                    if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                        q = img[i, j + 1]
                        r = img[i, j - 1]
                    # angle measure of 45
                    elif 22.5 <= angle[i, j] < 67.5:
                        q = img[i + 1, j - 1]
                        r = img[i - 1, j + 1]
                    # angle measure of 90
                    elif 67.5 <= angle[i, j] < 112.5:
                        q = img[i + 1, j]
                        r = img[i - 1, j]
                    # angle measure of 135
                    elif 112.5 <= angle[i, j] < 157.5:
                        q = img[i - 1, j - 1]
                        r = img[i + 1, j + 1]
                    # If pixel is white (255) keep value of pixel
                    if (img[i, j] >= q) and (img[i, j] >= r):
                        Z[i, j] = img[i, j]
                    else:  # if the pixels do not change in intensity then leave
                        Z[i, j] = 0
                except IndexError as e:
                    pass

        return Z

    def doubleThresh(self, img):
        highthreshratio = img.max() * self.high_thresh
        lowthreshratio = highthreshratio * self.low_thresh

        M, N = img.shape
        res = np.zeros((M, N), dtype=np.int32)

        weak = np.int32(self.weak_pixel)
        strong = np.int32(self.strg_pixel)

        strg_i, strg_j = np.where(img >= highthreshratio)
        zeros_i, zeros_j = np.where(img < lowthreshratio)

        weak_i, weak_j = np.where((img <= highthreshratio) & (img >= lowthreshratio))

        res[strg_i, strg_j] = strong
        res[weak_i, weak_j] = weak

        return res

    def hysteresis(self, img, strong=255):
        M, N = img.shape
        weak = self.weak_pixel

        for i in range(1, M - 1):
            for j in range(1, N - 1):
                if img[i, j] == weak:
                    try:
                        if ((img[i + 1, j - 1] == strong) or (img[i + 1, j] == strong) or (img[i + 1, j + 1] == strong)
                                or (img[i, j - 1] == strong) or (img[i, j + 1] == strong)
                                or (img[i - 1, j - 1] == strong) or (img[i - 1, j + 1] == strong) or (
                                        img[i - 1, j] == strong)):
                            img[i, j] = strong
                        else:
                            img[i, j] = 0
                    except IndexError as e:
                        pass
        return img

    def buildFileList(self, path):
        filenames = []
        dirct = []
        # loop through the directory to store the path for each image and the name of the image
        for root, subdirct, files in os.walk(path, topdown=True):
            for name in files:
                filenames.append(name)  # append file name to the end of the list
                dirct.append(os.path.join(root, name))  # append the full path and filename to the end of the list
        # print(filenames)    #view the list of images
        # print(dirct)    #view the list of fullpath and file name

        return filenames, dirct

    def edgeDetect(self, dirct):
        highlight_imgs = []
        for i, img in enumerate(dirct):
            Img = Image.open(img)  # using Image function from PIL library to hand TIFF file format
            imgarray = np.array(Img)
            self.applyBlur(imgarray)
            self.gradMat, self.thetaMat = self.sobelFilter(self.img_smooth)
            self.nonMaxImg = self.nonMaxSupress(self.gradMat, self.thetaMat)
            self.threshImg = self.doubleThresh(self.nonMaxImg)
            highlight_imgs = self.hysteresis(self.threshImg)
            self.highlight_imgs.append(highlight_imgs)
        return self.highlight_imgs
