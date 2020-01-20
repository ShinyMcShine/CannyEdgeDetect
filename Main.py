import EEG_CannyEdge as ced
import cv2 as cv
import matplotlib.pyplot as plt
from scipy.ndimage.filters import convolve
import numpy as np
from PIL import Image
#   Define Image(s) #
#imgpath = 'C:/Users/dcampoy/Pictures/Wallpapers/frank_frazetta_manape.jpg'
imgpath = 'C:/Users/dcampoy/Pictures/TestingCannyEdge/T1/IMG_137.tiff'

#img = cv.imread(imgpath, cv.IMREAD_GRAYSCALE)  # Using OpenCV function to read in image
img = Image.open(imgpath)   #using Image function from PIL libary to hand TIFF file format
#img.show()
imgarray = np.array(img)
#print(imgarray.size)
#print(imgarray.shape)

# Main #
path = 'C:/Users/dcampoy/Pictures/TestingCannyEdge/T0/'
ce = ced.CannyEdge()
# Currently I'm unable to pass the img variable to CannyEdge as says no arguments but should be fine passing only img var
ce.__int__(imgarray)    # This line is to call the __init__ initializer method to populate default params
# Load images #
filenames, dirct = ce.buildFileList(path)

# Blur the image using a kernel and sigma
blur_img = convolve(imgarray, ce.gausKrnl(ce.krnl_size, ce.sigma))

# Calculate gradients of blurred image to apply non-max-threshold algorithm
G, theta = ce.sobelFilter(blur_img)

# Apply non-max threshold to gradient image
nonmaxsupp_img = ce.nonMaxSupress(G, theta)

# using the non-max-supressed image the double threshold is applied
dblthresh_img = ce.doubleThresh(nonmaxsupp_img)

# created hysteresis to reinforce pixels that are closer to strong edges
hyst_img = ce.hysteresis(dblthresh_img)

#'''
# Plot Orig vs Blur
plt.subplot(1, 2, 1)
plt.imshow(ce.imgs, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(blur_img, cmap='gray')
plt.title('Blurred Image with Gaussian Kernel')

plt.show()
'''

'''
# Plot Blur vs SobelFilter
plt.subplot(1, 2, 1)
plt.imshow(blur_img, cmap='gray')
plt.title('Blurred Image')
gradients = G.astype('int32')   # converting float16 to int to plot image
plt.subplot(1, 2, 2)
plt.imshow(gradients, cmap='gray')
plt.title('Gradients from Image - Convolve with Sobel Filters ')

plt.show()
'''


'''
# Plot Sobelfilters vs NonMaxSuppression
gradients = G.astype('int32')   # converting float16 to int to plot image
plt.subplot(1, 2, 1)
plt.imshow(gradients, cmap='gray')
plt.title('Sobelfilters Image')

plt.subplot(1, 2, 2)
plt.imshow(nonmaxsupp_img, cmap='gray')
plt.title('NonMaxSuppression')

plt.show()
'''

'''
# Plot NonMaxSuppression vs Double Threshold
gradients = G.astype('int32')   # converting float16 to int to plot image
plt.subplot(1, 2, 1)
plt.imshow(nonmaxsupp_img, cmap='gray')
plt.title('NonMaxSuppression Image')

plt.subplot(1, 2, 2)
plt.imshow(dblthresh_img, cmap='gray')
plt.title('Double Threshold Image')

plt.show()
#'''

# Plot Double Threshold vs Hysteresis
plt.subplot(1, 2, 1)
plt.imshow(dblthresh_img, cmap='gray')
plt.title('Double Threshold Image')

plt.subplot(1, 2, 2)
plt.imshow(hyst_img, cmap='gray')
plt.title('Hysteresis Image')

plt.show()

plt.subplot(1, 2, 1)
plt.imshow(ce.imgs, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(hyst_img, cmap='gray')
plt.title('Hysteresis Image')
plt.show()

