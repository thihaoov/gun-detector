import cv2
import numpy as np

net = cv2.dnn.readNet('weights/yolov3_gun_final.weights', 'cfgs/yolov3_gun.cfg')
classes = []
with open('cfgs/gun.names', 'r') as f:
    classes = f.read().splitlines()

cap = cv2.VideoCapture(0)
# img = cv2.imread('image.jpg')

while True:
    _, frame = cap.read()
    cv2.normalize(frame, frame, 0, 255, cv2.NORM_MINMAX)

    height, width, _ = frame.shape

    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    # for b in blob:
    #     for n, img_blob in enumerate(b):
    #         cv2.imshow(str(n), img_blob)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []
    conf_threshold = 0.5

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x, center_y = int(detection[0] * width), int(detection[1] * height)
                w, h = int(detection[2] * width), int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    # print(len(boxes))
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, 0.4)
    # print(indexes.flatten())

    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(boxes), 3))

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = colors[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness=2)
            cv2.putText(frame, label + " " + confidence, (x, y-5), font, fontScale=2, color=color, thickness=2)

    cv2.imshow("Image", frame)
    key = cv2.waitKey(1)
    if key == 7:
        break

cap.release()
cv2.destroyAllWindows()
