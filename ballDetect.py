#! /usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from math import *
import time

cap = cv2.VideoCapture('video.3gp')
fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
writer = cv2.VideoWriter('output.avi', fourcc, 30.0, (1280, 720))

# ball tracking class
class CBall:
    ballPosition = {'x': 0, 'y': 0, 'r': 0}
    distance = 0.0

    countFit = 0

    distanceThreshold = 15
    repeatThreshold = 3

    def setPos(self, x, y, r):
        newDistance = sqrt(pow(x, 2) + pow(y, 2))
        if abs(self.distance - newDistance) < self.distanceThreshold:
            self.countFit += 1
            print 'addFit'
        else:
            self.countFit = 0
            print 'resetFit'

        # update position
        self.distance = newDistance
        self.ballPosition['x'], self.ballPosition['y'], self.ballPosition['r'] = x, y, r

        # is over limit
        if self.countFit > self.repeatThreshold:
            return True
        else:
            return False


ball = CBall()

tracker = None

while (True):
    ret, frame = cap.read()

    if ret:
        startTime = time.time()

        scale = 0.5

        inputImg = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

        HSVImg = cv2.cvtColor(inputImg, cv2.COLOR_RGB2HSV)

        thresholdImg = cv2.inRange(HSVImg[:, :, 0], 104, 112)

        blurImg = cv2.blur(thresholdImg, (7, 7))

        circles = cv2.HoughCircles(blurImg, cv2.HOUGH_GRADIENT, 2, 20, param1=400, param2=90)

        blurImg = cv2.cvtColor(blurImg, cv2.COLOR_GRAY2BGR)

        # draw circles
        if circles is not None:
            for (x, y, r) in circles[0, :]:
                if ball.setPos(x, y, r):
                    # update tracker
                    del tracker
                    tracker = cv2.TrackerKCF_create()
                    ok = tracker.init(inputImg, (int(x - r), int(y - r), int(r * 2), int(r * 2)))

                    cv2.rectangle(HSVImg, (int(x - r), int(y - r)), (int(x + r), int(y + r)),
                                  (200, 200, 200), 5, 1)

                cv2.circle(blurImg, (x, y), r, (200, 200, 200), 5)

        # if not ballFit:
        if tracker:
            ok, roi = tracker.update(inputImg)
            tracker.clear()
            if ok:
                p1 = (int(roi[0]), int(roi[1]))
                p2 = (int(roi[0]) + int(roi[2]), int(roi[1]) + int(roi[3]))
                cv2.rectangle(inputImg, p1, p2, (0, 200, 200), 5, 1)

        # hsvv = HSVImg[:, :, 0]
        h, s, v = cv2.split(HSVImg)
        hsvvn = cv2.cvtColor(h, cv2.COLOR_GRAY2BGR)

        imccn1 = np.concatenate((inputImg, HSVImg), axis=1)
        imccn2 = np.concatenate((hsvvn, blurImg), axis=1)
        imccn3 = np.concatenate((imccn1, imccn2), axis=0)

        cv2.imshow('imgOut', imccn3)

        writer.write(imccn3)
        cv2.waitKey(1)

        endTime = time.time()
        print 'time:', endTime - startTime

    else:
        break

cap.release()
writer.release()
