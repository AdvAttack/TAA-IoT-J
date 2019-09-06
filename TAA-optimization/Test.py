import os
import sys
import math  
import keras
import numpy as np 
np.set_printoptions(threshold=np.inf)
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
#from keras import optimizers
from scipy.misc import imread, imsave
from tensorflow.python.platform import flags
from cleverhans.utils_keras import conv_2d, cnn_model
from keras.models import load_model
from cleverhans.utils_tf import model_loss
#from utils26 import load_many_images


FLAGS = flags.FLAGS
flags.DEFINE_integer('nb_classes', 26, 'Number of classification classes') 
flags.DEFINE_integer('img_rows', 32, 'Input row dimension')
flags.DEFINE_integer('img_cols', 32, 'Input column dimension')
flags.DEFINE_integer('nb_channels', 3, 'Nb of color channels in the input.')
flags.DEFINE_string('checkpoint', 'train', 'Prefix to use when saving the checkpoint')
flags.DEFINE_string('model_path', './models/Adadelta-train-299-keras122', 'Path to load model from.')
flags.DEFINE_boolean('inverse_mask', False, 'Specifies whether to use an inverse mask (set all pixels in the original image to a specified value)')

def load_norm_img_from_source(src):
    '''
    read an image from the filepath specified
    and normalizes it to the [0.0,1.0] range
    :param src: the filepath to the image
    :return: the normalized image as a numpy array of shape (img_rows, img_cols, nb_channels)
    '''
#    img = cv2.imread(src)
    img = imread(src, mode='RGB')
    assert img is not None, "No image found at filepath %s"%src
    return img/255.0

def load_many_images(src):
    '''
    Loads all images in the specified folder
    by using load_img_inverse_mask
    SIDE EFFECT: prints the image names and the indices they were loaded to
    :param src: the path to the source directory to load
    :return: a Python array of the images loaded
    '''
    imgs = []
    filenames = os.listdir(src)
    for fname in filenames:
        if FLAGS.inverse_mask:
            imgs.append(load_img_inverse_mask(os.path.join(src, fname)))
        else:
            imgs.append(load_norm_img_from_source(os.path.join(src, fname)))
    print 'Loaded images in directory %s'%src
    map(lambda x: sys.stdout.write('Index %d image %s\n'%(x[0],x[1])), zip(range(len(filenames)), filenames))
    print 
    sys.stdout.flush()
    return imgs, filenames

def Normalization(x):
    return [(float(i)-min(x))/float(max(x)-min(x)) for i in x]


model = cnn_model(img_rows=FLAGS.img_rows, img_cols=FLAGS.img_cols, channels=FLAGS.nb_channels, nb_classes=FLAGS.nb_classes)
model.summary()

# Create TF session and set as Keras backend session
sess = tf.Session()
keras.backend.set_session(sess)
print("Created TensorFlow session and set Keras backend.")

saver = tf.train.Saver()
saver.restore(sess, FLAGS.model_path)
print("Loaded the parameters for the model from %s"%FLAGS.model_path)

# will hold the placeholders so that they can be returned 
placeholders = {}
img_rows = FLAGS.img_rows
img_cols = FLAGS.img_cols



# Test
imgs_test, filenames = load_many_images('./data/test/0')
noise = np.load("./optimization_output/Attention/noise_mul/noise_mul_Stop_SpeedLimit45.npy")


noise_imgs = []
total_testImg = len(imgs_test)
Testlabels = np.zeros((total_testImg,26), dtype=int)
for i in range(total_testImg):
	noise_imgs.append(sess.run(tf.clip_by_value(tf.add(imgs_test[i], noise),0,1)))
	Testlabels[i,0] = 1 

placeholders['image_test'] = tf.placeholder(tf.float32, shape = (None, img_rows, img_cols, FLAGS.nb_channels))
placeholders['True_test_labels'] = tf.placeholder(tf.float32, shape = (None, FLAGS.nb_classes))
varops = {}
varops['pred_test'] = model(placeholders['image_test'])

feed_test = {placeholders['image_test']: noise_imgs, 
             placeholders['True_test_labels']: Testlabels, 
             keras.backend.learning_phase(): 0}

prediction_test = sess.run( \
    (varops['pred_test']) \
    , feed_dict=feed_test)


num_misclassified = 0
for j in range(total_testImg):
	GroundTruth_test = np.argmax(Testlabels[j])
	ModelPred_test = np.argmax(prediction_test[j])
	if GroundTruth_test != ModelPred_test and ModelPred_test == 11:
		num_misclassified += 1

proportion_misclassified = float(num_misclassified)/float(total_testImg)
print 'Targeted attack accuracy is %.1f'%(proportion_misclassified*100.0)
print 


'''
num_correct_test = 0
for j in range(total_testImg):
	GroundTruth_test = np.argmax(Testlabels[j])
	ModelPred_test = np.argmax(prediction_test[j])
	if GroundTruth_test == ModelPred_test:
		num_correct_test += 1

proportion_correct_test = float(num_correct_test)/float(total_testImg)
print 'Test Accuracy is %.1f'%(proportion_correct_test*100.0)
print 
'''

sess.close()












