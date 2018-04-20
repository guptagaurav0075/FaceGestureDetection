import os
import cv2
from PIL import *

PATH = os.getcwd()
PATH = PATH.rstrip("/src")
fileName = PATH + "/Dataset/label/label.lst"
LABELS_FILE = file(fileName, "r")
labels = LABELS_FILE.readlines()
LABELS_FILE.close();
updatedDataset = PATH+"/dataset_updated"
updatedImageCounter = 0
# for expression label
# "0" "angry"
# "1" "disgust"
# "2" "fear"
# "3" "happy"
# "4" "sad"
# "5" "surprise"
# "6" "neutral"

expression_label_count = {
    '0':0,
    '1':0,
    '2':0,
    '3':0,
    '4':0,
    '5':0,
    '6':0
}
expression_label_desc = {
    '0':"angry.jpg",
    '1':"disgust.jpg",
    '2':"fear.jpg",
    '3':"happy.jpg",
    '4':"sad.jpg",
    '5':"surprise.jpg",
    '6':"neutral.jpg"
}

def printExpressionLableCount():
    count = 0
    global expression_label_count
    for i in expression_label_count:
        print i," --> ", expression_label_count[i]


def checkAndUpdateExpressionCount(val):
    global expression_label_count
    val  = val.strip()
    data = val.split(" ")
    if(len(data)!=8 or float(data[6]) <80):
        # print val
        return
    count = expression_label_count[data[7]]
    count+=1
    expression_label_count[data[7]] = count

def checkifValidData(val):
    val = val.strip()
    data = val.split(" ")
    if (len(data) != 8 or float(data[6]) < 80):
        return False
    return True


def readAndUpdateDataset(val):
    if(not checkifValidData(val)):
        return
    updatePicture(val)

def updatePicture(val):
    global updatedImageCounter, updatedDataset, expression_label_desc
    print val
    # image_name, face_id_in_image, face_box_top, face_box_left, face_box_right, face_box_bottom, face_box_cofidence, expression_label
    val = val.strip()
    data = val.split(" ")
    updatedImage = updatedDataset + "/" + str(updatedImageCounter) + "_" +expression_label_desc[data[7]]
    updatedImageCounter+=1
    img_path = PATH+"/Dataset/Img/"+data[0]
    img = cv2.imread(img_path)
    y = int(data[2])
    h = int(data[5])-y
    x = int(data[3])
    w = int(data[4])-x

    crop_img = img[y:y+h, x:x+w]
    crop_img = cv2.resize(crop_img,(100,100))
    cv2.imwrite(updatedImage, crop_img)



count = 0;
printExpressionLableCount()
for i in labels:
    checkAndUpdateExpressionCount(i)
    readAndUpdateDataset(i)
    # print count, "  --->   ", i
    count+=1
printExpressionLableCount()
