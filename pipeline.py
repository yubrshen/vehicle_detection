def slide_windows(focused_width, focused_height, h_overlap = 0.3, v_overlap = 0.3):
    """
    Returns a list of bounding box coordinates, relative to the focused_image_area,
    defined by the focused_width, and focused_height.
    """
    rate = 1/2                  # used to specify the window sections
    window_specification_per_v_position = [  # list of window specifications, as a list of
        # the window width, the window height, the application section specifications:
        # the vertical window start, vertical window end, and
        # the horizontal window start, window horizontal end
        # all the coordinates are relative to the focused_image_area

        [200, 170, int(focused_height*rate), focused_height,  # from focused_height*(1/2) to focused_height # 240, 160
         0, focused_width, 0.5, 0.3], # lower
        [150, 120, int(focused_height*rate*6 / 4), focused_height-int(focused_height*rate*rate*rate),  # 150, 120
         0, focused_width, 0.5, 0.3], # lower-middle
        [120, 110, int(focused_height*rate*6 / 4), focused_height-int(focused_height*rate*rate), # 120, 110
         0, focused_width, 0.9, 0.3], # middle
        [120, 96, int(focused_height*rate*rate*rate), focused_height-int(focused_height*rate), # 120, 96
         0, focused_width, 0.9, 0.3], # upper-middle
        [90, 60, 0, int(focused_height*rate*rate), # 90, 60 # original: 84, 64, need to enlarge to 90, and 60 to catch the side view cars
         0, focused_width, 0.9, 0.3] # top
    ]
    window_list = []
    for window_specification in window_specification_per_v_position:
        h_window_size, v_window_size, v_window_start, v_window_stop, h_window_start, h_window_stop, h_overlap, v_overlap = window_specification
        # The number of pixels shift in horizontal or vertical for sliding windows
        nh_pix_per_step = np.int(h_window_size*(1 - h_overlap))
        nv_pix_per_step = np.int(v_window_size*(1 - v_overlap))

        # number of sliding windows in horizontal or vertical direction:
        nh_windows = max(1, np.int((h_window_stop - h_window_start) / nh_pix_per_step))
        # at least one scanned window
        nv_windows = max(1, np.int((v_window_stop - v_window_start) / nv_pix_per_step))
        assert ((0 < nh_windows) and (0 < nv_windows)), (window_specification, nh_windows, nv_windows, "Zero number of window!")
        for vertical_index in range(nv_windows):
            for horizontal_index in range(nh_windows):
                start_h = horizontal_index * nh_pix_per_step + h_window_start
                end_h = start_h + h_window_size
                start_v = vertical_index * nv_pix_per_step + v_window_start
                end_v = start_v + v_window_size
                window_list.append(((start_h, start_v), (end_h, end_v)))
    return window_list

import utils
from sklearn.preprocessing import StandardScaler
def find_cars(img, svc, X_scaler, focused_vertical_start=400, focused_vertical_end=640,
              orientation=8, pix_per_cell=8, cell_per_block=2,
              spatial_size=16, hist_bins=32, color_space='HLS', sample_window=64,
              channels=[0], sliding_windows=None,
              debug=False, fname=None, frame_index=None):
    # Must have the normalization, otherwise, there will be excessive false positive. 
    img = img.astype(np.float32)/255  # this normalization turned out to be very needed
    # for accuracy, and generalization

    img_tosearch = img[focused_vertical_start:focused_vertical_end,:,:] 
    # only focus on the region of possibilities
    if sliding_windows is None:
        sliding_windows = slide_windows(img_tosearch.shape[1], img_tosearch.shape[0],
                                        h_overlap=0.9, v_overlap=0.5)
    # End of if len(sliding_windows)

    if color_space != ident_config['default_color_space']:
        ctrans_tosearch = cv2.cvtColor(
            img_tosearch,
            eval('cv2.COLOR_' + ident_config['default_color_space'] + '2' + color_space))
    else:
        ctrans_tosearch = img_tosearch
    # End of if color_space
    bboxes = []
    draw_img = np.copy(img)
    frame_index = frame_index or 0
    for window in sliding_windows:
        ((start_h, start_v), (end_h, end_v)) = window
        # Extract the image patch
        subimg = cv2.resize(
            ctrans_tosearch[start_v:end_v, start_h:end_h], (sample_window, sample_window))
        # Get color features
        spatial_features, hist_features = utils.color_features(
            subimg, spatial_size=spatial_size, hist_bins=hist_bins, channels=channels)
        hog_features= utils.get_hog_features_channels(
            subimg, orientation, pix_per_cell, cell_per_block, channels)
        # Scale features and make a prediction
        test_features = np.hstack((spatial_features, hist_features, hog_features))
        # test_features = hog_features
        test_features = test_features.reshape(1, -1)
        test_features = X_scaler.transform(test_features)
        test_prediction = svc.predict(test_features)

        focused_horizontal_start = 0
        confidence = svc.decision_function(test_features)[0]
        if (debug and frame_index < 0) or ((test_prediction == 1) and (0.3 < confidence)):
            top_left = (start_h + focused_horizontal_start, start_v + focused_vertical_start)
            bottom_right = (end_h + focused_horizontal_start, end_v + focused_vertical_start)
            box_color = (0,0,255) if (test_prediction==1) else (200, 200, 0)
            line_width = 3 if (test_prediction==1) else 1
            cv2.rectangle(draw_img, top_left, bottom_right, (0, 0, 255), line_width)
            if debug:
                mpimg.imsave('./output_images/boxed_' + str(frame_index) + '_' + utils.image_capture_name(fname), draw_img)
            # End of debug

            if ((test_prediction == 1) and (0.3 < confidence)):
                bboxes.append((top_left, bottom_right))
            # End of if ((test_prediction == 1) and (0.3 < confidence))

        # End of if (debug and frame_index < 10) or ((test_prediction == 1) and (0.3 < confidence))
    # End of for window in sliding_windows
    return bboxes, draw_img, sliding_windows

import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os.path
from scipy.ndimage.measurements import label
def compute_heatmap(initial_heatmap, bboxes, history, threshold=6):
    heatmap_accumulated = []
    heatmap = initial_heatmap
    for i in range(len(history)):
        if history[i]:
            heatmap = add_heat(heatmap, history[i], value=(2 if (i < 2) else 1))
        # End of if history...
        heatmap_accumulated.append(heatmap)
    # End of for i in range(len(history))...
    if bboxes:
        heatmap = add_heat(heatmap, bboxes, value = 2)
    # End of if bboxes
    heatmap_accumulated[0] = heatmap

    heatmap, accumulated = apply_threshold(heatmap, threshold=threshold)
    return heatmap, heatmap_accumulated          # the heatmmap can be used for label

def shift_history(history, recent_bboxes):
    for i in range(len(history)-1, 0, -1):  # reverse
        history[i] = history[i-1]
    # End of for in ...
    history[0] = recent_bboxes
    return history
from utils import image_capture_name

def show_cars(img, bboxes, history=None, threshold=1, debug=False, fname=None, frame_index=None):
    initial_heatmap = np.zeros_like(img[:,:,0])
    heatmap, heatmap_accumulated = compute_heatmap(initial_heatmap, bboxes, history, threshold=threshold)

    # Find final boxes from heat using label function
    labels = label(heatmap)
    draw_img, derived_bboxes = draw_labeled_bboxes(np.copy(img), labels, heatmap_threshold=heatmap)
    # to show the nonzero min, maximum in heatmap
    history = shift_history(history, derived_bboxes)
    frame_index = frame_index or 0

    if debug and frame_index < 0: # True and 0 < np.max(heatmap):
        fig = plt.figure()
        plt.subplot(131)
        plt.imshow(draw_img)
        plt.title('Car Positions')
        plt.subplot(132)
        plt.imshow(heatmap, cmap='hot')
        plt.title('Heat Map')
        fig.tight_layout()
        fig.savefig("./output_images/cars_" + str(frame_index) + "_" + image_capture_name(fname))
        plt.show()
        plt.close()
    # End of if debug
    if debug and frame_index == 23:
        # print out history heatmaps
        num_heatmaps = len(history)
        num = num_heatmaps + 2
        import matplotlib.gridspec as gridspec
        columns = 1; rows = num // columns
        figure = plt.figure(figsize=(columns*4, rows*2))
        gs1 = gridspec.GridSpec(rows, columns)
        gs1.update(wspace=0.9, hspace=0.9) # set the spacing between axes.
        row_height = 2; columns_width = 3
        for j in range(rows):
            ax1 = plt.subplot(gs1[j, 0])
            ax1.set_xticklabels([]); ax1.set_yticklabels([])
            ax1.set_aspect('equal')
            if j < rows - 2:
                title = "History Heatmap Accumulated at -" + str(rows - j) + "th Detection"
                plt.imshow(heatmap_accumulated[j], cmap='hot')
            elif j == rows - 2:
                title = "Current Heatmap"
                plt.imshow(heatmap, cmap='hot')
            else:
                title = "Current Vehicle Dectection"
                plt.imshow(draw_img)
            # End of if j < row - 2
            ax1.set_title(title)
        # End of for j in range(rows)
        figure.savefig(
            "./output_images/history_heatmap" + str(frame_index) + "_" + image_capture_name(fname) )
        plt.show()
        plt.close()
    # End of if debug and frame_index == 9
    return draw_img, history, derived_bboxes

def add_heat(heatmap, bbox_list, value=1):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Accumulate += confidence for all pixels inside each bbox
        ((start_h, start_v), (end_h, end_v)) = box
        heatmap[start_v:end_v, start_h:end_h] += value
    return heatmap

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    to_display = np.copy(heatmap)
    to_display[heatmap <= threshold] = 0
    heatmap[0 < heatmap] = heatmap[0 < heatmap] - 50
    # Return thresholded map
    return to_display, heatmap

def draw_labeled_bboxes(img, labels, heatmap_threshold=None):
    # Iterate through all detected cars
    bboxes = []
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
        bboxes.append(bbox)
    # End of for car_number
    if heatmap_threshold is not None:
        nonzero = heatmap_threshold[0 < heatmap_threshold]
        if 0 < nonzero.size:
            min_nonzero = np.min(nonzero)
            max_nonzero = np.max(nonzero)
            mean_nonzero = np.mean(nonzero)
            # draw text to show the nonzero confidence.
            txt_display = [ "The highest confidence: " + str(max_nonzero),
                            "The lowest confidence: " + str(min_nonzero),
                            "The mean of confidence: " + str(mean_nonzero) ]
            offset = 0
            for txt in txt_display:
                cv2.putText(
                    img, txt, (50, 70 + offset*27), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                offset += 1
            # End of for txt...
        # End of if 0 < ...
    # Return the image
    return img, bboxes

import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os.path
from ident_config import ident_config
import pickle
import glob
import utils
from ident_config import ident_config

def pipeline(img, svc, X_scaler, history=None, threshold=1, sliding_windows=None, debug=False, fname=None, frame_index=None):
    # global heatmap 
    bboxes, draw_img, sliding_windows = find_cars(
        img, svc, X_scaler, focused_vertical_start=ident_config['v_start'], focused_vertical_end=ident_config['v_stop'],
        orientation=ident_config['orientation'],
        pix_per_cell=ident_config['pix_per_cell'], cell_per_block=ident_config['cells_per_block'],
        spatial_size=ident_config['spatial_size'], hist_bins=ident_config['hist_bins'],
        color_space=ident_config['color_space'], sample_window=ident_config['sample_window'],
        channels=eval(ident_config['channels']), sliding_windows=sliding_windows,
        debug=debug, fname=fname, frame_index=frame_index)
    draw_img, history, derived_bboxes = show_cars(
        img, bboxes, history=history, threshold=threshold, debug=debug, fname=fname, frame_index=frame_index)

    return draw_img, history, sliding_windows, derived_bboxes

def main():
    trained_pickle = pickle.load( open("trained_pickle.p", "rb" ) )
    svc = trained_pickle["svc"]
    X_scaler = trained_pickle["scaler"]

    fname = './test_images/test1.jpg'
    img = utils.imread_scaled_unified(fname)
    heatmap = np.zeros_like(img[:,:,0])
    sliding_windows = None
    fname = None
    paths = [fname] if fname else glob.glob("./test_images/*.jpg")
    for p in paths:
        history = [None, None, None, None] 
        img = utils.imread_scaled_unified(p)
        draw_img, history, sliding_windows, derived_bboxes = pipeline(
            img, svc, X_scaler, history=history, threshold=9,
            sliding_windows=sliding_windows, debug=True, fname=p)
        heatmap = np.zeros_like(img[:,:,0])
        # reset for single image processing
        # display done in show in pipeline
    print('pipeline done')
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
