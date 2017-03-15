from pipeline import pipeline
import pickle
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os.path
from moviepy.editor import VideoFileClip

history = [None, None, None, None]  # persist across images
sliding_windows = None
frame_index = 0

def main():
    trained_pickle = pickle.load( open("trained_pickle.p", "rb" ) )
    svc = trained_pickle["svc"]
    X_scaler = trained_pickle["scaler"]

    fname = "./test_video.mp4"
    #fname = "./project_video.mp4"

    def video_process(img):
        global history
        #heatmap_video,
        global sliding_windows
        global frame_index
        draw_img, history, sliding_windows, derived_bboxes = pipeline(
            img, svc, X_scaler, history=history, threshold=8,
            sliding_windows=sliding_windows, debug=True, fname=fname, frame_index=frame_index)
        frame_index += 1

        return draw_img

    clip = VideoFileClip(fname)
    output_clip = clip.fl_image(video_process)
    output_clip.write_videofile("./output_images/marked_" + os.path.basename(fname), audio=False)

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
