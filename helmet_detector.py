#!/usr/bin/env python

import cv2
import numpy as np

debug = True


def helmet_detector(roi):
    filtered_roi = cv2.GaussianBlur(roi, (5,5),0)
    hsv = cv2.cvtColor(filtered_roi, cv2.COLOR_BGR2HSV)
    #hue_sum = np.sum(hsv[0]/(hsv.shape[0]*hsv.shape[1]))
    hsv_mean = np.mean(hsv[:, :, 0])
    print(hsv_mean)


if __name__ == '__main__':

    sample_fname = '/home/al/projects/helmet_detector/samples/16474950.mp4'
#    capture = cv2.VideoCapture(0)
    capture = cv2.VideoCapture(sample_fname)

    while True:
        ok, image = capture.read()
#        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        full_path2cascade = '/home/al/github/opencv/data/haarcascades/haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(full_path2cascade)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            #roi = img[y:y+h, x:x+w]
            #TODO: run color detector on extended roi
            helmet_height_fract = 0.2
            helmet_height = int(h*helmet_height_fract)
            roi_helmet = ((x, y), (x+w, y-helmet_height))
            if debug:
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.rectangle(image, roi_helmet[0], roi_helmet[1], (0, 0, 255), 1)
            roi = image[x:x+w, y-helmet_height:y, :]
            helmet_detector(roi)

        if ok:
            cv2.imshow('Helmet_detector', image)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        else:
           print('Reading image source error.')
           break
