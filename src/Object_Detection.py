# Importing Required Module for Project

import numpy as np
from ultralytics import YOLO
import cv2 as vision
import cvzone
from sort import * // https: // github.com/abewley/sort

# Source of Object Collection in Video File

CaptureMoment = vision.VideoCapture('../Custom_Objects.mp4')

# Model Creation which downloads required Weights automatically

Project_Model = YOLO('yolov8l.pt') // Large Model

Store_Classes = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
                 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'zebra', 'giraffe', 'backpack', 'umbrella', 'tie']
