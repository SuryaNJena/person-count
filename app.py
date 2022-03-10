import cv2
import math
import numpy as np
from flask import Flask, jsonify
import threading

app = Flask(__name__)

# Open video file or capture device.
video = cv2.VideoCapture('video.mp4')

confidence_thr = 0.5

#Load the Caffe model
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt", "MobileNetSSD_deploy.caffemodel")

persons = 0
exit_count = 0
counts = {'exit_count': 0}
trackers = []

FINISH_LINE = 350 # persons will be counted when hitting this line
START_LINE = 100  # persons won't be added until beyond the line

# OpenCV colors are (B, G, R) tuples
WHITE = (255, 255, 255)
YELLOW = (66, 244, 238)
GREEN = (80, 220, 60)
LIGHT_CYAN = (255, 255, 224)
DARK_BLUE = (139, 0, 0)
GRAY = (128, 128, 128)

def add_new_object(obj, image, persons):
    person = str(persons)
    xmin = obj[0]
    xmax = obj[1]
    ymin = obj[2]
    ymax = obj[3]

    # init tracker
    tracker = cv2.legacy.TrackerKCF_create()
    tracker.init(image, (xmin, ymin, xmax-xmin, ymax-ymin))
    trackers.append((tracker, person))

def update_trackers(image):
    tboxes = []
    color = (80, 220, 60)
    fontface = cv2.FONT_HERSHEY_SIMPLEX
    fontscale = 1
    thickness = 1
    for n, pair in enumerate(trackers):
        tracker, person = pair
        textsize, _baseline = cv2.getTextSize(person, fontface, fontscale, thickness)
        success, bbox = tracker.update(image)

        if not success:
            del trackers[n]
            continue

        tboxes.append(bbox)  # Return updated box list

        xmin = int(bbox[0])
        ymin = int(bbox[1])
        xmax = int(bbox[0] + bbox[2])
        ymax = int(bbox[1] + bbox[3])
        xmid = int(round((xmin+xmax)/2))
        ymid = int(round((ymin+ymax)/2))

        global entry_count,exit_count
        if ymid <= START_LINE:
            # Stop tracking persons when they hit finish line
            exit_count+=1
            print('Someone Exited')
            del trackers[n]
        elif ymid >= FINISH_LINE:
            del trackers[n]
        else:
            # Rectangle and number on the persons we are tracking
            label_object(color, YELLOW, fontface, image, person, textsize, 4, xmax, xmid, xmin, ymax, ymid, ymin)

    return tboxes

def not_tracked(objects, utboxes):
    if not np.any(objects):
        return []  # No new classified objects to search for
    if not np.any(utboxes):
        return objects  # No existing boxes, return all objects

    new_objects = []
    for obj in objects:
        ymin = obj[2]
        ymax = obj[3]
        ymid = int(round((ymin+ymax)/2))
        xmin = obj[0]
        xmax = obj[1]
        xmid = int(round((xmin+xmax)/2))
        box_range = ((xmax - xmin) + (ymax - ymin))/4
        for bbox in utboxes:
            bxmin = int(bbox[0])
            bymin = int(bbox[1])
            bxmax = int(bbox[0] + bbox[2])
            bymax = int(bbox[1] + bbox[3])
            bxmid = int((bxmin + bxmax) / 2)
            bymid = int((bymin + bymax) / 2)
            if math.sqrt((xmid - bxmid)**2 + (ymid - bymid)**2) < box_range:
                # found existing, so break (do not add to new_objects)
                break
        else:
            new_objects.append(obj)

    return new_objects

def label_object(color, textcolor, fontface, image, person, textsize, thickness, xmax, xmid, xmin, ymax, ymid, ymin):
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, thickness)
    pos = (xmid - textsize[0]//2, ymid + textsize[1]//2)
    cv2.circle(image, pos, 5, textcolor, -1)

def in_range(obj):
    xmin = obj[0]
    xmax = obj[1]
    ymin = obj[2]
    ymax = obj[3]
    if ymin < START_LINE or ymax > FINISH_LINE:
        # Don't add new trackers before start or after finish.
        # Start line can help avoid overlaps and tracker loss.
        # Finish line protection avoids counting the person twice.
        return False
    return True

def main(cap):
    while True:
        detected_persons = []
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == False:
            break
        
        # reduce frame size to speed up inference
        #frame = cv2.resize(frame,(int(frame.shape[1]/1.5),int(frame.shape[0]/1.5)))
        frame_resized = cv2.resize(frame,(300,300)) # resize frame for prediction

        # MobileNet requires fixed dimensions for input image(s)
        # so we have to ensure that it is resized to 300x300 pixels.
        # set a scale factor to image because network the objects has differents size.
        # We perform a mean subtraction (127.5, 127.5, 127.5) to normalize the input;
        # after executing this command our "blob" now has the shape:
        # (1, 3, 300, 300)
        blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
        #Set to network the input blob
        net.setInput(blob)
        #Prediction of network
        detections = net.forward()

        #Size of frame resize (300x300)
        cols = frame_resized.shape[1]
        rows = frame_resized.shape[0]

        #For get the class and location of object detected,
        # There is a fix index for class, location and confidence
        # value in @detections array .
        for i in range(detections.shape[2]):
            class_id = int(detections[0, 0, i, 1]) # Class label
            confidence = detections[0, 0, i, 2] #Confidence of prediction
            # Filter prediction and 15 is class identifier number for persons/humans 
            if class_id == 15 and confidence > confidence_thr:
                # Object location 
                xLeftBottom = int(detections[0, 0, i, 3] * cols)
                yLeftBottom = int(detections[0, 0, i, 4] * rows)
                xRightTop   = int(detections[0, 0, i, 5] * cols)
                yRightTop   = int(detections[0, 0, i, 6] * rows)
                
                # Factor for scale to original size of frame
                heightFactor = frame.shape[0]/300.0
                widthFactor = frame.shape[1]/300.0

                # Scale object detection to frame
                xLeftBottom = int(widthFactor * xLeftBottom)
                yLeftBottom = int(heightFactor * yLeftBottom)
                xRightTop   = int(widthFactor * xRightTop)
                yRightTop   = int(heightFactor * yRightTop)

                detected_persons.append((xLeftBottom,xRightTop,yLeftBottom,yRightTop))

                # Label and start tracking
                boxes = update_trackers(frame)
                not_tracked_persons = not_tracked(detected_persons, boxes)
                for obj in not_tracked_persons:
                    if in_range(obj):
                        global persons
                        persons+=1
                        add_new_object(obj, frame, persons)

        cv2.line(frame, (0,START_LINE), (frame.shape[1],START_LINE), (255,0,0), 2)
        cv2.line(frame, (0,FINISH_LINE), (frame.shape[1],FINISH_LINE), (0,0,255), 2)
        cv2.rectangle(frame, (0, 5), (125, 30), (0,0,0), -1)
        cv2.putText(frame, "Exit Count="+str(exit_count), (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0),2)
        cv2.putText(frame, "Exit Count="+str(exit_count), (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),1)

        #cv2.imshow("CAM_FEED", frame)

        if cv2.waitKey(1) >= 0:
            cv2.destroyAllWindows() 
            break

        counts['exit_count'] = exit_count

@app.route('/',methods=['GET'])
def home():
    return '<p>API link is <a href="/get_count">/get_count</a></p>\n'

@app.route('/get_count',methods=['GET'])
def get():
    return jsonify(counts)

p1 = threading.Thread(target=main,args=(video,))
p1.start()

if __name__ == "__main__":
    p2 = threading.Thread(target=app.run,args=())
    p2.start()
