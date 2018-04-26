import tensorflow as tf
import numpy
from helperFunctions import *



INPUT_DATA_SET = []
OUTPUT_DATA_SET = []
PATH = os.getcwd()
PATH = PATH.rstrip("/src")
PATH = PATH+"/dataset_updated"
MODEL_FILE = PATH+"/models_v1/model"

generate_input_vector(INPUT_DATA_SET=INPUT_DATA_SET, OUTPUT_DATA_SET=OUTPUT_DATA_SET,TRAIN_FOLDER_NAME= PATH)


print "INPUT DATASET :",len(INPUT_DATA_SET)
print "OUTPUT DATASET :",len(OUTPUT_DATA_SET)

randomize_Data(INPUT_DATA_SET, OUTPUT_DATA_SET)

TRAIN_INPUT = []
TRAIN_OUTPUT = []
TEST_INPUT = []
TEST_OUTPUT = []

#SPLIT_PERCENTAGE will decide the input split between input and output, currently it is 80-20%
SPLIT_PERCENTAGE = 80

split_Data(INPUT_DATASET=INPUT_DATA_SET, OUTPUT_DATASET=OUTPUT_DATA_SET, TRAIN_INPUT=TRAIN_INPUT, TRAIN_OUTPUT=TRAIN_OUTPUT, TEST_INPUT=TEST_INPUT, TEST_OUTPUT=TEST_OUTPUT, SPLIT_PERCENTAGE=SPLIT_PERCENTAGE)

print("INPUT DATASET :",len(INPUT_DATA_SET))
print("OUTPUT DATASET :",len(OUTPUT_DATA_SET))

print("Train Input: ", len(TRAIN_INPUT))
print("Train Output: ", len(TRAIN_OUTPUT))

print("TEST Input: ", len(TEST_INPUT))
print("TEST Output: ", len(TEST_OUTPUT))



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



def train_neural_network(x):
    global MODEL_FILE
    prediction,regularizers = conv_neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    regularizers_cost = cost + regularization_alpha*regularizers

    optimizer = tf.train.AdamOptimizer().minimize(regularizers_cost)

    hm_epochs = 10
    saver = tf.train.Saver()
    print("\n\n\nStarting Training")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            print("\n Epoch : ", epoch, " is Starting")
            randomize_Data(TRAIN_INPUT, TRAIN_OUTPUT);
            epoch_loss = 0
            iterations = int(len(TRAIN_INPUT) / batch_size)-1
            for i in range(iterations):
                epoch_x, epoch_y = getNextBatch(TRAIN_INPUT,TRAIN_OUTPUT,batch_size,i)
                sess.run(optimizer, feed_dict={x: epoch_x, y: epoch_y})
                c = sess.run(cost, feed_dict={x:epoch_x, y:epoch_y})
                print("\t\titeration  -->",i,"/",iterations,"  cost --> ",c, " total epoch loss --> ", epoch_loss)
                epoch_loss += c

            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            # randomize_Data(INPUT_DATA_SET, OUTPUT_DATA_SET)
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            test_accuracy = accuracy.eval({x: TEST_INPUT, y: TEST_OUTPUT})
            train_accuracy = accuracy.eval({x: TRAIN_INPUT, y: TRAIN_OUTPUT})
            print('Training Accuracy: ', train_accuracy )
            print ('Testing Accuracy: ', test_accuracy)
            print("\n")

        saver.save(sess=sess, save_path=MODEL_FILE)
        sess.close()


train_neural_network(x)
