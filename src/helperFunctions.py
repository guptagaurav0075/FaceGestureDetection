import os
import numpy
from PIL import Image
import random

# for expression label
# "0" "angry"
# "1" "disgust"
# "2" "fear"
# "3" "happy"
# "4" "sad"
# "5" "surprise"
# "6" "neutral"


__EXPRESSION_LABEL_DESCRIPTION = {
    'angry': [1, 0, 0, 0, 0, 0, 0],
    'disgust': [0, 1, 0, 0, 0, 0, 0],
    'fear': [0, 0, 1, 0, 0, 0, 0],
    'happy': [0, 0, 0, 1, 0, 0, 0],
    'sad': [0, 0, 0, 0, 1, 0, 0],
    'surprise': [0, 0, 0, 0, 0, 1, 0],
    'neutral': [0, 0, 0, 0, 0, 0, 1],
}

def getNextBatch(input,output,batch_size,i):
    epochX = [];
    epochxY = [];
    startPoint = i*batch_size
    for j in range(batch_size):
        epochX.append(input[startPoint+j]);
        epochxY.append((output[startPoint+j]));
    return epochX,epochxY;


def split_Data(INPUT_DATASET, OUTPUT_DATASET, TRAIN_INPUT, TRAIN_OUTPUT, TEST_INPUT, TEST_OUTPUT, SPLIT_PERCENTAGE):
    SPLIT_PERCENTAGE *= 0.01
    maxTrain = len(INPUT_DATASET)*SPLIT_PERCENTAGE
    maxTrain = int(maxTrain)
    for i in range(maxTrain):
        TRAIN_INPUT.append(INPUT_DATASET[i])
        TRAIN_OUTPUT.append(OUTPUT_DATASET[i])
    for i in range(maxTrain, len(INPUT_DATASET)):
        TEST_INPUT.append(INPUT_DATASET[i])
        TEST_OUTPUT.append(OUTPUT_DATASET[i])

def randomize_Data(arrA, arrB):
    for i in range(len(arrA)-1):
        # print "i ->", i+1,"   len(arrA)-1 --> ",len(arrA) - 1
        j = random.randint(i + 1, len(arrA) - 1)
        swap(i, j, arrA, arrB)

def swap(i, j, arr_a, arr_b):
    temp =arr_a[i]
    arr_a[i] = arr_a[j]
    arr_a[j] = temp

    temp = arr_b[i]
    arr_b[i] = arr_b[j]
    arr_b[j]= temp

def generate_input_vector(INPUT_DATA_SET, OUTPUT_DATA_SET, TRAIN_FOLDER_NAME):
    """This Function generated the input data set by reading the files in TRAIN_FOLDER_NAME"""
    if os.path.exists(TRAIN_FOLDER_NAME):
        for f in os.listdir(TRAIN_FOLDER_NAME):
            if f.endswith('.jpg'):
                numpy_array = numpy.array(Image.open(TRAIN_FOLDER_NAME + '/' + f).convert('1'))
                INPUT_DATA_SET.append(numpy.invert(numpy_array.flatten()))

                if f.endswith('angry.jpg'):
                    OUTPUT_DATA_SET.append(__EXPRESSION_LABEL_DESCRIPTION["angry"]);
                elif f.endswith('disgust.jpg'):
                    OUTPUT_DATA_SET.append(__EXPRESSION_LABEL_DESCRIPTION['disgust']);
                elif f.endswith('fear.jpg'):
                    OUTPUT_DATA_SET.append(__EXPRESSION_LABEL_DESCRIPTION['fear']);
                elif f.endswith('happy.jpg'):
                    OUTPUT_DATA_SET.append(__EXPRESSION_LABEL_DESCRIPTION['happy']);
                elif f.endswith('sad.jpg'):
                    OUTPUT_DATA_SET.append(__EXPRESSION_LABEL_DESCRIPTION['sad']);
                elif f.endswith('surprise.jpg'):
                    OUTPUT_DATA_SET.append(__EXPRESSION_LABEL_DESCRIPTION['surprise']);
                elif f.endswith('neutral.jpg'):
                    OUTPUT_DATA_SET.append(__EXPRESSION_LABEL_DESCRIPTION['neutral']);
