#!/usr/bin/env python

import cv2
import numpy as np
import argparse
import copy

ESCAPE_KEY = 27
FRAMERATE = 33


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'file',
        help='file to parse')
    args = parser.parse_args()
    return args

SCALING = 1
DEFAULT_WINDOW_SIZE=(640*SCALING,480*SCALING)

def main():
    args = parse_args()
    print(cv2.__version__)

    # Handler for video frames
    cap = cv2.VideoCapture(args.file)

    # Trained XML classifiers describes some features of some object we want to detect
    car_cascade = cv2.CascadeClassifier('cars.xml')

    fgbg = cv2.createBackgroundSubtractorMOG2()

    old_gray, p0, lk_params,mask, color  = setup_optical_flow(cap)

    # loop runs if capturing has been initialized.
    while True:
        # reads frame from a video
        ret, frame = cap.read()
        framecopy = copy.deepcopy(frame)
        gray = cv2.cvtColor(framecopy, cv2.COLOR_BGR2GRAY)

        # bg subtraction
        fgmask = fgbg.apply(framecopy)

        # Detects cars of different sizes in the input image
        cars = car_cascade.detectMultiScale(gray, 1.1, 1)

        img = calc_optical_flow(old_gray, gray, p0, lk_params, mask, color, framecopy)

        frame_area(img)

        # To draw a rectangle in each cars
        for (x,y,w,h) in cars:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

        cv2.namedWindow( "Display window", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO);
        cv2.resizeWindow('Display window', *DEFAULT_WINDOW_SIZE)
        cv2.moveWindow('Display window', 0,0)
        cv2.namedWindow( "Optical Flow", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO);
        cv2.resizeWindow('Optical Flow', * DEFAULT_WINDOW_SIZE)
        cv2.moveWindow('Optical Flow', DEFAULT_WINDOW_SIZE[0],0)
        cv2.namedWindow( "BG", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO);
        cv2.resizeWindow('BG', *DEFAULT_WINDOW_SIZE)
        cv2.moveWindow('BG', 2*DEFAULT_WINDOW_SIZE[0],0)


        # Display frame in a window
        cv2.imshow('Display window', frame)
        cv2.imshow('Optical Flow',img)
        cv2.imshow('BG',fgmask)

        # Wait for Esc key to stop
        if cv2.waitKey(FRAMERATE) == ESCAPE_KEY:
            break

    # De-allocate any associated memory usage
    cv2.destroyAllWindows()
    cap.release()

def setup_optical_flow(cap):
    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                           qualityLevel = 0.3,
                           minDistance = 7,
                           blockSize = 7 )
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    # Create some random colors
    color = np.random.randint(0,255,(100,3))
    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)
    return old_gray, p0, lk_params, mask, color

def calc_optical_flow(old_gray, gray, p0, lk_params, mask, color, frame):
    # return gray
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, p0, None, **lk_params)
    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]
    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
    img = cv2.add(frame,mask)
    # Now update the previous frame and previous points
    old_gray = gray.copy()
    p0 = good_new.reshape(-1,1,2)
    return img


def frame_area(frame):
    x=70
    y=130
    w=150
    h=75
    # Draw a rectangle in the detection area
    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)


if __name__ == '__main__':
    main()
