import os
import math  
#import keras
from tensorflow import keras
import numpy as np 
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
#from keras import optimizers
from scipy.misc import imread
from tensorflow.python.platform import flags
from cleverhans.utils_keras import conv_2d, cnn_model
from keras.models import load_model
from cleverhans.utils_tf import model_loss
from tensorflow.core.protobuf import saver_pb2

FLAGS = flags.FLAGS
flags.DEFINE_integer('nb_classes', 26, 'Number of classification classes') 
flags.DEFINE_integer('img_rows', 32, 'Input row dimension')
flags.DEFINE_integer('img_cols', 32, 'Input column dimension')
flags.DEFINE_integer('nb_channels', 3, 'Nb of color channels in the input.')
flags.DEFINE_string('checkpoint', 'Adadelta-train', 'Prefix to use when saving the checkpoint')


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

#Generate list
classOrder = []
label_class = []

#s1 read all the images under the path, and give labels
def get_files(ratio):
	for k in range(26):
		classOrder.append(k)
		label_class.append(k)
		classO = []
		label_c = []
		for file in os.listdir('./data/' + str(k)):
			classO.append('./data/' + str(k) + '/' + file) 
			classOrder[k] = classO
			label_c.append(k)
			label_class[k] = label_c
     
#s2 random order
	image_list=np.hstack((classOrder[0], classOrder[1], classOrder[2], classOrder[3], classOrder[4], classOrder[5], classOrder[6], classOrder[7], classOrder[8], classOrder[9], classOrder[10], classOrder[11], classOrder[12], classOrder[13], classOrder[14], classOrder[15], classOrder[16], classOrder[17], classOrder[18], classOrder[19], classOrder[20], classOrder[21], classOrder[22], classOrder[23], classOrder[24], classOrder[25]))
	label_list=np.hstack((label_class[0], label_class[1], label_class[2], label_class[3], label_class[4], label_class[5], label_class[6], label_class[7], label_class[8], label_class[9], label_class[10], label_class[11], label_class[12], label_class[13], label_class[14], label_class[15], label_class[16], label_class[17], label_class[18], label_class[19], label_class[20], label_class[21], label_class[22], label_class[23], label_class[24], label_class[25]))
    #shuffle
	temp = np.array([image_list, label_list])
	temp = temp.transpose()
	np.random.shuffle(temp)
    #convert img and lab to list
	all_image_list=list(temp[:,0])
	all_label_list=list(temp[:,1])
    #divide the List to train and val, ratio is the proportion of val
	n_sample = len(all_label_list)  
	n_val = int(math.ceil(n_sample*ratio))   #val samples
	n_train = n_sample - n_val   #train samples
    
	tra_images = all_image_list[0:n_train]
	tra_labels = all_label_list[0:n_train]  
	tra_labels = [int(float(i)) for i in tra_labels]  
	val_images = all_image_list[n_train:]  
	val_labels = all_label_list[n_train:]
	val_labels = [int(float(i)) for i in val_labels]    
	return tra_images,tra_labels,val_images,val_labels

tra_images,tra_labels,val_images,val_labels = get_files(0.2)

imgs = []
for fname in tra_images:
	imgs.append(load_norm_img_from_source(fname))

# Convert the images and labels from list to array
TrnLabels = np.array(tra_labels)
Img = np.array(imgs)

labels = np.zeros((6004,26), dtype=int)
for i in range(6004):
	labels[i,TrnLabels[i]] = 1

# Define model
model = cnn_model(img_rows=FLAGS.img_rows, img_cols=FLAGS.img_cols, channels=FLAGS.nb_channels, nb_classes=FLAGS.nb_classes)
model.summary()

# will hold the placeholders so that they can be returned 
placeholders = {}
img_rows = FLAGS.img_rows
img_cols = FLAGS.img_cols
placeholders['image_in'] = tf.placeholder(tf.float32, shape = (None, img_rows, img_cols, FLAGS.nb_channels))
placeholders['True_labels'] = tf.placeholder(tf.float32, shape = (None, FLAGS.nb_classes))

# will hold the variables and operations defined from now on
varops = {}
varops['pred'] = model(placeholders['image_in'])
varops['loss'] = model_loss(placeholders['True_labels'], varops['pred'], mean=True)

feed_dict = {placeholders['image_in']: imgs, 
             placeholders['True_labels']: labels, 
             keras.backend.learning_phase(): 0}


# Create TF session and set as Keras backend session
sess = tf.Session()
keras.backend.set_session(sess)
print("Created TensorFlow session and set Keras backend.")

#op = tf.train.AdamOptimizer(learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False,name='Adam').minimize(varops['loss'])
op = tf.train.AdadeltaOptimizer(learning_rate=0.1, rho=0.95, epsilon=1e-08, use_locking=False,name='Adadelta').minimize(varops['loss'])
#op = tf.train.AdagradOptimizer(learning_rate=0.01, initial_accumulator_value=0.1, use_locking=False,name='Adagrad').minimize(varops['loss'])

sess.run(tf.global_variables_initializer())

saver = tf.train.Saver(max_to_keep=10, write_version=saver_pb2.SaverDef.V2)

for i in range(300):
	print 'Epoch %d'%i,
	
	_, train_loss, prediction = sess.run( \
	    (op, \
	    varops['loss'], \
	    varops['pred']) \
	    , feed_dict=feed_dict)
	
	num_correct = 0
	
	for j in range(6004):
		GroundTruth = np.argmax(labels[j])
		ModelPred = np.argmax(prediction[j])
		if GroundTruth == ModelPred:
			num_correct += 1
	
	proportion_correct = float(num_correct)/float(6004)
	print 'percent correct classified images %.1f'%(proportion_correct*100.0)
	print 
	
	saver.save(sess, os.path.join('train_checkpoint_Adadelta', FLAGS.checkpoint), global_step=i, write_state=True)

# Test
imgs_test = []
for fname_test in val_images:
	imgs_test.append(load_norm_img_from_source(fname_test))

TestLabels = np.array(val_labels)
Testlabels = np.zeros((1501,26), dtype=int)
for i in range(1501):
	Testlabels[i,TestLabels[i]] = 1

placeholders['image_test'] = tf.placeholder(tf.float32, shape = (None, img_rows, img_cols, FLAGS.nb_channels))
placeholders['True_test_labels'] = tf.placeholder(tf.float32, shape = (None, FLAGS.nb_classes))
varops['pred_test'] = model(placeholders['image_test'])
feed_test = {placeholders['image_test']: imgs_test, 
             placeholders['True_test_labels']: Testlabels, 
             keras.backend.learning_phase(): 0}

prediction_test = sess.run( \
    (varops['pred_test']) \
    , feed_dict=feed_test)

num_correct_test = 0

for j in range(1501):
	GroundTruth_test = np.argmax(Testlabels[j])
	ModelPred_test = np.argmax(prediction_test[j])
	if GroundTruth_test == ModelPred_test:
		num_correct_test += 1
	
proportion_correct_test = float(num_correct_test)/float(1501)
print 'Test Accuracy is %.1f'%(proportion_correct_test*100.0)
print 


sess.close()



