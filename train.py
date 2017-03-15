import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os.path
def samples_sorted():
    """
    returns the paths of the sample images for cars and non-cars,
    including duplication of some samples.
    """

    import glob
    cars_original = glob.glob("./vehicles/*/*.png")
    # The following are duplicated:
    cars_KITTI = glob.glob("./vehicles/KITTI_extracted/*.png")
    cars_GTI_Right = glob.glob("./vehicles/GTI_Right/*.png")
    cars_GTI_Left = glob.glob("./vehicles/GTI_Left/*.png")
    cars = cars_original + cars_KITTI + cars_GTI_Left + cars_GTI_Right
    # The above introduces duplication of samples, causing bleeding of training samples into validation
    np.random.shuffle(cars)     # side effect return None
    cars_to_be_augmented = cars_GTI_Left + cars_GTI_Right
    np.random.shuffle(cars_to_be_augmented)
    num_cars = len(cars) + len(cars_to_be_augmented)

    non_cars_original = glob.glob("./non-vehicles/*/*.png")
    # The following are duplicated:
    non_cars_Extras = glob.glob("./non-vehicles/Extras/*.png")
    noncars = non_cars_original + non_cars_Extras + non_cars_Extras
    # The above introduces duplication of samples, causing bleeding of training samples into validation
    np.random.shuffle(noncars)  # side effect return None
    num_noncars = len(noncars)
    return cars, noncars, cars_to_be_augmented, num_cars, num_noncars

def show_samples(num=1):
    import matplotlib.image as mpimg
    from utils import show_pair
    from utils import get_hog_features
    from ident_config import ident_config

    cars, noncars, cars_to_be_augmented, num_cars, num_noncars = samples_sorted()
    cars_to_show = [ mpimg.imread(car) for car in cars[:num] ]
    noncars_to_show = [ mpimg.imread(noncar) for noncar in noncars[:num] ]
    show_pair(cars_to_show, noncars_to_show, title="samples")

    # Show HOG
    def change_color_space(to_be):
        # change the color space
        color_space = ident_config["color_space"]
        converted = (
            to_be if (color_space == ident_config["default_color_space"]) else [
                cv2.cvtColor(image,
                             eval('cv2.COLOR_' + ident_config['default_color_space'] + '2' + color_space))
                for image in to_be])
        return converted

    num_to_exam = min(num, 5)

    cars_for_features = change_color_space(cars_to_show[:num_to_exam])
    noncars_for_features = change_color_space(noncars_to_show[:num_to_exam])

    channels= eval(ident_config['channels'])

    def hog_feature_images(to_be):
        # list of (per channel)
        # list of (per image)
        # hog_images
        hog_images = []
        for channel in channels:
            per_channels = []
            for image in to_be:
                _, hot_image = utils.get_hog_features(image[:,:,channel], ident_config['orientation'], ident_config['pix_per_cell'], ident_config['cells_per_block'], vis=True)
                per_channels.append(hot_image)
            # End of for image in to_be
            hog_images.append(per_channels)
        # End of for channel in channels
        return hog_images

    car_hog_features = hog_feature_images(cars_for_features)
    noncar_hog_features = hog_feature_images(noncars_for_features)
    # per channel,
    for channel in channels:
       show_pair(cars_to_show[:num_to_exam], car_hog_features[channel], title="cars_hog_ch" + str(channel))
       show_pair(noncars_to_show[:num_to_exam], noncar_hog_features[channel], title="noncars_hog_ch" + str(channel))

    # spatial features
    spatial_size = ident_config['spatial_size']
    car_spatial_features = [
        cv2.resize(image, (spatial_size, spatial_size)) for image in cars_for_features]
    show_pair(cars_to_show[:num_to_exam], car_spatial_features, title="car_spatial")
    noncar_spatial_features = [
        cv2.resize(image, (spatial_size, spatial_size)) for image in noncars_for_features]
    show_pair(noncars_to_show[:num_to_exam], noncar_spatial_features, title="noncar_spatial")

    # Color histograms
    hist_bins=ident_config['hist_bins']
    car_hist = [ np.concatenate(
        tuple(np.histogram(image[:,:,i], bins=hist_bins)[0] for i in channels))
                 for image in cars_for_features]
    show_pair(
        cars_to_show[:num_to_exam], car_hist, title="hist_cars", even_hist=True)
    noncar_hist = [ np.concatenate(
        tuple(np.histogram(image[:,:,i], bins=hist_bins)[0] for i in channels))
                 for image in noncars_for_features]
    show_pair(
        noncars_to_show[:num_to_exam], noncar_hist, title="hist_noncars", even_hist=True)

import utils
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split

def training_features(orientation=8, pix_per_cell=8, cell_per_block=2,
                      spatial_size=16, hist_bins=32, color_space='HLS', sample_window=64,
                      channels=[0], debug=False):
    """
    from the file paths of cars, and noncars
    extract features and labels for training and validation.
    """
    def extract(paths, augment=False):         # extract and augment
        features = []
        for file in paths:
            image = utils.imread_scaled_unified(file)
            if color_space != ident_config['default_color_space']:
                image_color_converted = cv2.cvtColor(
                    image,
                    eval('cv2.COLOR_' + ident_config['default_color_space'] + '2' + color_space))
            else:
                image_color_converted = image
            # End of if color_space

            image_resized = cv2.resize(image_color_converted, (sample_window, sample_window))
            if augment:
                brightened = utils.brighten(image_resized, bright=1.2)
                flipped = cv2.flip(utils.brighten(image_resized, bright=1.1), 1)  # horizontal flip
                to_process = [brightened, flipped]
            else:
                to_process = [image_resized]
            # End of if augment

            for x in to_process: # must use square bracket for single element in list to iterate
                # using tuple, it will iterate the single image's row dimension. 
                hog_features = utils.get_hog_features_channels(
                    x, orientation, pix_per_cell, cell_per_block, channels)
                spatial_features, hist_features = utils.color_features(
                    x, spatial_size=spatial_size, hist_bins=hist_bins, channels=channels)
                image_features = np.hstack(
                    (spatial_features, hist_features, hog_features)).reshape(1, -1)
                image_features = np.squeeze(image_features)
                # remove the redundant dimension, StandardScaler does not like it
                features.append(image_features)
            # End of for x ...
        # End of for file
        return features
    cars, noncars, cars_to_be_augmented, num_cars, num_noncars = samples_sorted()
    num_samples = 30000         # limit the number of samples to be selected from each group.
    print('num_cars: ', num_cars, ' num_noncars: ', num_noncars, ' max. samples: ', 3*num_samples)

    car_features = extract(cars[:min(num_samples, len(cars))], augment=False)
    car_augmented_features = extract(cars_to_be_augmented[:min(num_samples, len(cars_to_be_augmented))], augment=True)
    noncar_features = extract(noncars[:min(num_samples, len(noncars))], augment=False)

    # Create an array stack of feature vectors
    X = np.vstack((car_features, car_augmented_features, noncar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)
    del X                       # X, scaled_X consumes much memory, should be released ASAP.
    # Define the labels vector
    y = np.hstack((np.ones(len(car_features) + len(car_augmented_features)), np.zeros(len(noncar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.1, random_state=rand_state)
    return X_train, X_test, y_train, y_test, X_scaler

import time
from sklearn.svm import LinearSVC
from ident_config import ident_config

def train_svc(X_train, X_test, y_train, y_test):
    # Use a linear SVC 
    svc = LinearSVC(C=ident_config['C_SVM'])
    # Check the training time for the SVC
    t=time.time()
    num_samples = 50000
    svc.fit(X_train[:min(num_samples, y_train.size)], y_train[:min(num_samples, y_train.size)])
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    test_accuracy = round(svc.score(X_test, y_test), 4)
    print('Test Accuracy of SVC = ', test_accuracy)
    train_accuracy = round(svc.score(X_train, y_train), 4)
    print('Train Accuracy of SVC = ', train_accuracy)
    # Check the prediction time for a single sample
    t=time.time()
    n_predict = 10
    print('SVC predicts for testing samples:\t', svc.predict(X_test[0:n_predict]))
    print('For these', n_predict, 'labels:\t\t\t', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

    print('SVC predicts for training samples:\t', svc.predict(X_train[0:n_predict]))
    print('For these', n_predict, 'labels:\t\t\t', y_train[0:n_predict])

    return svc, test_accuracy

def svc_pickle():
    from ident_config import ident_config
    from print_dict import print_dict
    import pickle
    import os.path

    print('parameters: ')
    print_dict(ident_config)

    X_train, X_test, y_train, y_test, X_scaler = training_features(
        orientation=ident_config['orientation'],
        pix_per_cell=ident_config['pix_per_cell'], cell_per_block=ident_config['cells_per_block'],
        spatial_size=ident_config['spatial_size'], hist_bins=ident_config['hist_bins'],
        color_space=ident_config['color_space'], sample_window=ident_config['sample_window'],
        channels=eval(ident_config['channels']), debug=False)

    svc, accuracy = train_svc(X_train, X_test, y_train, y_test)

    # TODO: report training performance

    # pickle training results
    trained_pickle = {}
    trained_pickle['svc'] = svc
    trained_pickle['scaler'] = X_scaler
    pickle.dump(trained_pickle, open("./trained_pickle.p", "wb"))

    print('accuracy: ', accuracy)
    return svc, accuracy

def main():
    svc_pickle()
    return 0

# imports
import sys
# constants

# exception classes

# interface functions

# classes

# internal functions & classes

if __name__ == '__main__':
    status = main()
    sys.exit(status)
