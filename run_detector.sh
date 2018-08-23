#!/bin/bash

python3 ./helmet_detector.py ./samples/helmet.mp4 ./cascades/haarcascade_frontalface_alt2.xml
python3 ./helmet_detector.py ./samples/no_helmet.mp4 ./cascades/haarcascade_frontalface_alt2.xml
