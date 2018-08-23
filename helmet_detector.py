#!/usr/bin/env python3

import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("input", help="Input video file.")
parser.add_argument("haar_cascade", help="Opencv haar face detection cascade.")
args = parser.parse_args()

debug = True


def detect_helmet(roi):
    filtered_roi = cv2.GaussianBlur(roi, (5,5),0)
    hsv = cv2.cvtColor(filtered_roi, cv2.COLOR_BGR2HSV)
    hsv_mean = np.mean(hsv[:, :, 0])
    if hsv_mean > 10 and hsv_mean < 20:
        return True
    else:
        return False


if __name__ == '__main__':

    sample_fname = args.input
    full_path2cascade = args.haar_cascade

    capture = cv2.VideoCapture(sample_fname)

    while True:
        ret, image = capture.read()
        face_cascade = cv2.CascadeClassifier(full_path2cascade)
        faces = face_cascade.detectMultiScale(image, 1.3, 5)
        for (x, y, w, h) in faces:
            helmet_height_fract = 0.3
            helmet_height = int(h*helmet_height_fract)
            roi_helmet = ((x, y), (x+w, y-helmet_height))
            if debug:
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.rectangle(image, roi_helmet[0], roi_helmet[1], (0, 0, 255), 1)
            roi = image[y-helmet_height:y, x:x+w :]
            text_shift = 20
            text_origin = (x, y + w + text_shift)
            if detect_helmet(roi) == True:
                cv2.putText(image, 'Helmet detected', text_origin,
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 140, 255), 2)
            else:
                cv2.putText(image, 'Alert: no helmet', text_origin,
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if ret == True:
            cv2.imshow('Helmet_detector', image)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        else:
           print('Reading image source error.')
           break
