from skimage.feature import hog
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                     vis=False, feature_vec=True):
    """
    Given a one channel img, calculate its HOG feature.
    """
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=False, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=False, 
                       visualise=vis, feature_vector=feature_vec)
        return features

def get_hog_features_channels(img, orientation, pix_per_cell, cell_per_block, channels):
    """
    Given a list of color channels, compute the HGO features for each of the channels, and
    concatenate the features horizontally.
    """
    hog_features = np.hstack(tuple(get_hog_features(
        img[:,:,i], orientation, pix_per_cell, cell_per_block, feature_vec=False).ravel()
                                   for i in channels))
    return hog_features

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

def show_pair(odd, even, title="samples.jpg", even_hist=False):
    num = len(odd)*2
    columns = 5
    rows = num // columns
    figure = plt.figure()
    gs1 = gridspec.GridSpec(rows, columns)
    gs1.update(wspace=0.9, hspace=0.9) # set the spacing between axes.
    row_height = 2
    columns_width = 3
    plt.figure(figsize=(columns*columns_width, rows*row_height))
    for j in range(rows):
        for i in range(columns):
            ax1 = plt.subplot(gs1[j, i])
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
            ax1.set_aspect('equal')
            ax1.set_title(title)
            if j % 2 == 0:
                k = (j // 2)
                plt.imshow(odd[i+k*columns])
            elif j % 2 == 1:
                k = (j-1) // 2
                if even_hist:
                    hist_value = even[i+k*columns]
                    xticks_pos = range(1, len(hist_value)+1)
                    plt.axis('tight')
                    plt.margins(0.05, 0)
                    plt.xlim(1, 96)  # must set xlim, and ylim to have meaningful bar chart
                    plt.ylim(0, max(hist_value))
                    plt.bar(xticks_pos, hist_value, align='center') # columns_width/len(hist_value)*100
                else:
                    plt.imshow(even[i+k*columns])
            # End of if j
        # End of for i
    # End of for j
    plt.savefig("./output_images/" + title + ".jpg") # savefig must be before show(), 
    # otherwise savefig will just save a blank picture
    plt.show()

import matplotlib.image as mpimg
def imread_scaled_unified(fname):
    """
    """
    img = mpimg.imread(fname)
    # from os.path import splitext
    # proper, ext = splitext(fname)
    # if ext == '.png':
    #     return 255*img
    # else:
    #     return img
    return img

import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os.path
def color_features(image, spatial_size=16, hist_bins=32, channels=[0]):
    spatial_features = cv2.resize(image, (spatial_size, spatial_size)).ravel()
    hist_features = np.concatenate(
        tuple(np.histogram(image[:,:,i], bins=hist_bins)[0] for i in channels))
    return spatial_features, hist_features

def brighten(image, bright=1.25):
    """Apply brightness conversion to RGB image

       # Args
           image(ndarray): RGB image (3-dimension)
           bright(float) : bright value for multiple
       # Returns
           image(ndarray): RGB image (3-dimension)
    """
    toHSV_be_brightened = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    toHSV_be_brightened[:,:,2] = toHSV_be_brightened[:,:,2]*bright
    return cv2.cvtColor(toHSV_be_brightened, cv2.COLOR_HSV2RGB)

def image_capture_name(fname):
    from os.path import splitext
    proper, ext = splitext(os.path.basename(fname))
    return proper + ".jpg"
