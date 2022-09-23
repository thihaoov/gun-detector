import cv2
import numpy as np

net = cv2.dnn.readNet('weights/yolov3_gun_final.weights', 'cfgs/yolov3_gun.cfg')
classes = {}
with open('cfgs/gun.names','r') as f:
    classes = f.read().splitlines()

img = cv2.imread('test_data/shotgun.jpg')
height, width, _ = img.shape

# returns a 4-dimensional array/blob
blob = cv2.dnn.blobFromImage(img, scalefactor=1/255, size=(416, 416), mean=(0, 0, 0), swapRB=True, crop=False)

for b in blob:
    for n, img_blob in enumerate(b):
        cv2.imshow(str(n), img_blob)

net.setInput(blob)

output_layers_names = net.getUnconnectedOutLayersNames()
layerOutputs = net.forward(output_layers_names)
print(len(layerOutputs))

boxes = []
confidences = []
class_ids = []

for output in layerOutputs:
    for detection in output:
        # each detection is an array of 85 values, 
        # the first 4 values are location indexes of bounding boxes, x, y, width and height
        # the other are scores of multiclass detection
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)

            w = int(detection[2] * width)
            h = int(detection[3] * height)

            x = int(center_x - w/2)
            y = int(center_y - h/2)

            boxes.append([x, y, w, h])
            confidences.append((float(confidence)))
            class_ids.append(class_id)

# Take indexes of objects' bounding boxes, params are score and nms thresholds 0.5 and 0.4 
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4) # Non Maximum Suppression Algorithm

font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(len(boxes), 3)) # array of random color values

for i in indexes.flatten():
    x, y, w, h = boxes[i]
    label = str(classes[class_ids[i]])
    confidence = str(round(confidences[i], 2))
    color = colors[i]
    cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
    cv2.putText(img, label + " " + confidence, (x, y-5), font, 1, color, 2)

cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()