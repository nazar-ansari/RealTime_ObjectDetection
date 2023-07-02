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

# custom area crop using photoshop for implementing specific area in result

Intersect = vision.imread('../co-ordinated_Segment_photoshop.img')

amb_tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

Counting_Line_Coordinate = [400, 297, 673, 297]

Total_No_Appearance = []

while True:
    success, Image = CaptureMoment.read()
    Main_Region = vision.bitwise_and(Image, Intersect)
    App_Logo = vision.imread('../Title_Logo.png', vision.IMREAD_UNCHANGED)
    Final_Image = cvzone.overlayPNG(Image, App_Logo, (0, 0))
    Result = Project_Model(Main_Region, stream=True)

    # to store unique detection
    Detection_List = np.empty((0, 5))

    for sub_part in Result:
        Detected_Box = sub_part.boxes
        for each_box in Detected_Box:
            co_x1, co_y1, co_x2, co_y2 = each_box.xyxy[0]
            co_x1, co_y1, co_x2, co_y2 = int(co_x1), int(
                co_y1), int(co_x2), int(co_y2)
            width, height = co_x2-co_x1, co_y2-co_y1
            Detection_Accuracy = math.ceil((each_box.conf[0])*100) / 100
            Class_Index = each_box.cls[0]
            Class_Belong = Store_Classes[Class_Index]

            if Class_Belong in ['car', 'bus', 'truck', 'motorbike'] and Detection_Accuracy > 0.3:
                Current_Value = np.array(
                    [co_x1, co_y1, co_x2, co_y2, Detection_Accuracy])
                Detection_List = np.vstack((Detection_List, Current_Value))

    Result_Tracked_Value = amb_tracker.update(Detection_List)
    vision.line(Image, (Counting_Line_Coordinate[0], Counting_Line_Coordinate[1]), (
        Counting_Line_Coordinate[2], Counting_Line_Coordinate[3]), (0, 0, 255), 5)

    for Update_Result in Result_Tracked_Value:
        co_x1, co_y1, co_x2, co_y2, Generated_ID = Updated_Result
        co_x1, co_y1, co_x2, co_y2 = int(co_x1), int(
            co_y1), int(co_x2), int(co_y2)
        print(Updated_Result)
        width, height = co_x2-co_x1, co_y2-co_y1
        cvzone.cornerRect(Image, (co_x1, co_y1, width, height),
                          l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(Image, f'{int(Generated_ID)}', (max(
            0, co_x1), max(35, co_y1)), scale=2, thickness=3, offset=10)
        center_co_x, center_co_y = co_x1+width//2, co_y1+height//2
        vision.circle(Image, (center_co_x, center_co_y),
                      5, (255, 0, 255), vision.FILLED)

        if Counting_Line_Coordinate[0] < center_co_x < Counting_Line_Coordinate[2] and Counting_Line_Coordinate[1]-15 < center_co_y < Counting_Line_Coordinate[3]+15:
            if Total_No_Appearance.count(Generated_ID) == 0:
                Total_No_Appearance.append(Generated_ID)
            vision.line(Image, (Counting_Line_Coordinate[0], Counting_Line_Coordinate[1]), (
                Counting_Line_Coordinate[2], Counting_Line_Coordinate[3]), (0, 255, 0), 5)

    vision.putText(Image, str(len(Total_No_Appearance)), (255, 100),
                   vision.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 8)

    vision.imshow("Image", Image)

    vision.waitKey(1)
