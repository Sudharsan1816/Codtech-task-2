import cv2
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,
                help = 'path to input image')
ap.add_argument('-c', '--config', required=True,
                help = 'path to yolo config file')
ap.add_argument('-w', '--weights', required=True,
                help = 'path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', required=True,
                help = 'path to text file containing class names')
args = ap.parse_args()


def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    try:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    
image = cv2.imread(args.image)

Width = image.shape[1]
Height = image.shape[0]
#  normalise the pixel value by 1/255
scale = 0.00392
classes = None
# this line of code safely opens a specified file for reading, allows to access 
# its contents while ensuring that the file is properly closed after we done using it
with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]
# generates random floating-point numbers in a specified range. Here, it generates 
# numbers between 0 and 255, which corresponds to the RGB color space (Red, Green, Blue).
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
# it loads a pre-trained deep learning model into our program using OpenCV's 
# Deep Neural Network (DNN) module.
net = cv2.dnn.readNet(args.weights, args.config)
 
blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

net.setInput(blob)
# processes the input data
outs = net.forward(get_output_layers(net))
#  initializes an empty list to store  of detected objects
class_ids = []
confidences = []
boxes = []
# detections with confidence scores above this threshold 
conf_threshold = 0.5
nms_threshold = 0.4

# loop iterates over each output from the neural network
for out in outs:
    for detection in out:
        # extracts the confidence scores for each class from the detection
        scores = detection[5:]
        # find the index of the class with the highest score
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            #  x-coordinate of the center of the bounding box 
            center_x = int(detection[0] * Width)
            center_y = int(detection[1] * Height)
            w = int(detection[2] * Width)
            h = int(detection[3] * Height)
            x = center_x - w / 2
            y = center_y - h / 2
            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])


indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
# loop iterates over indices, which contains the indices of the bounding boxes
for i in indices:
    try:
        box = boxes[i]
    except:
        i = i[0]
        box = boxes[i]
    
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]
    draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
# display the image in the window
cv2.imshow("object detection", image)
cv2.waitKey()
    
cv2.imwrite("object-detection.jpg", image)
cv2.destroyAllWindows()