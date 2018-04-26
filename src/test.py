import tensorflow as tf
import numpy
from helperFunctions import *
import datetime



TEST_INPUT_DATA_SET = []
TEST_OUTPUT_DATA_SET = []
PATH = os.getcwd()
PATH = PATH.rstrip("/src")
PATH = PATH+"/dataset_updated"
MODEL_FILE = PATH+"/models_v1/model"

generate_input_vector(INPUT_DATA_SET=TEST_INPUT_DATA_SET, OUTPUT_DATA_SET=TEST_OUTPUT_DATA_SET, TRAIN_FOLDER_NAME= PATH)


print "INPUT DATASET :",len(TEST_INPUT_DATA_SET)
print "OUTPUT DATASET :",len(TEST_OUTPUT_DATA_SET)

n_classes = 7
batch_size = 1000



x = tf.placeholder('float', [None, 100*100])
y = tf.placeholder('float', [None, n_classes])
regularization_alpha = 1





def conv2D(x,W):
    return tf.nn.conv2d(x,W, strides=[1,5,5,1], padding='SAME')

def maxpool2D(x):
    return tf.nn.max_pool(x, ksize=[1,5,5,1], strides=[1,5,5,1], padding='SAME')


def conv_neural_network_model(x):
    regularization = 0
    weights = {
        'conv1': tf.Variable(tf.random_normal([5,5,1,150])),
        'conv2': tf.Variable(tf.random_normal([10,10,150,200])),
        'conv3': tf.Variable(tf.random_normal([15,15,200,500])),
        'fc': tf.Variable(tf.random_normal([500,1024])),
        'out': tf.Variable(tf.random_normal([1024,n_classes]))
    }
    biases = {
        'conv1': tf.Variable(tf.random_normal([150])),
        'conv2': tf.Variable(tf.random_normal([200])),
        'conv3': tf.Variable(tf.random_normal([500])),
        'fc': tf.Variable(tf.random_normal([1024])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }
    x = tf.reshape(x, shape=[-1,100,100,1])
    conv1 = conv2D(x, weights['conv1'])
    conv1 = tf.nn.relu(conv1)+biases['conv1']
    conv1 = maxpool2D(conv1)
    regularization += tf.nn.l2_loss(weights['conv1'])

    conv2 = conv2D(conv1, weights['conv2'])
    conv2 = tf.nn.relu(conv2)+biases['conv2']
    conv2 = maxpool2D(conv2)
    regularization += tf.nn.l2_loss(weights['conv2'])

    conv3 = conv2D(conv2, weights['conv3'])
    conv3 = tf.nn.relu(conv3)+biases['conv3']
    conv3 = maxpool2D(conv3)
    regularization += tf.nn.l2_loss(weights['conv3'])

    fc = tf.reshape(conv3, [-1,500])
    fc = tf.nn.relu(tf.matmul(fc, weights['fc'])+biases['fc'])
    regularization += tf.nn.l2_loss(weights['fc'])

    output = tf.matmul(fc, weights['out'])+biases['out']

    return output, regularization




def test():
    global MODEL_FILE

    print "Building network"
    prediction, regularizers = conv_neural_network_model(x)
    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    print "Network built"

    with tf.Session() as session:
        saver = tf.train.Saver()
        saver.restore(session, MODEL_FILE)
        print "Model restored"
        print "Start testing"
        acc = accuracy.eval({x: TEST_INPUT_DATA_SET, y: TEST_OUTPUT_DATA_SET})
        print "Testing Complete"
        print "Overall Accuracy:", acc


test()