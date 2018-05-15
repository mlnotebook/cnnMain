from __future__ import division

import argparse
import sys
import matplotlib.pyplot as plt
import numpy as np
import h5py as h5py
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.layers import (Input, Activation, advanced_activations, Conv3D, Conv1D, Dense, Dropout, Flatten, MaxPooling3D, GlobalAveragePooling2D)
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.losses import binary_crossentropy, categorical_crossentropy
from keras.models import Sequential, load_model, Model
from keras.optimizers import Adam, SGD
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from callbacks import TensorBoard, ModelCheckpoint # custom Keras callback.py. Adds per-batch metrics for tensorboard
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.constraints import max_norm
from keras.regularizers import l2
from keras import backend as K

from scipy.stats import wasserstein_distance

''' MY SCRIPTS '''
from infuncs3 import get_split, get_batch
from ResNet3D import Resnet3DBuilder

''' Set some options in TensorFlow that will allow better memory usage of the GPU '''
from keras.backend.tensorflow_backend import set_session
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95, allow_growth=True)

''' OPTIONS TO PREVENT LEAKING OF PROCESSESS TO OTHER CPUS ON CLUSTER'''
cluster = False
if cluster:
	inter_op_parallelism_threads=8
	intra_op_parallelism_threads=8
	config = tf.ConfigProto(gpu_options=gpu_options, inter_op_parallelism_threads=inter_op_parallelism_threads, intra_op_parallelism_threads=intra_op_parallelism_threads)
else:
	config = tf.ConfigProto(gpu_options=gpu_options)

### START THE TENSORFLOW SESSION
set_session(tf.Session(config=config))


def plot_history(history, result_dir):
	### FUNCTION: Plots the graphs for accuracy and loss at the end of training
	# history: a Kears history obeject
	# result_dir: where to store the plots

	plt.plot(history.history['acc'], marker='.')
	plt.plot(history.history['val_acc'], marker='.')
	plt.title('model accuracy')
	plt.xlabel('epoch')
	plt.ylabel('mean_absolute_error')
	plt.grid()
	plt.legend(['mean_absolute_error', 'val_mean_absolute_error'], loc='lower right')
	plt.savefig(os.path.join(result_dir, 'model_accuracy.png'))
	plt.close()

	plt.plot(history.history['loss'], marker='.')
	plt.plot(history.history['val_loss'], marker='.')
	plt.title('model loss')
	plt.xlabel('epoch')
	plt.ylabel('loss')
	plt.grid()
	plt.legend(['loss', 'val_loss'], loc='upper right')
	plt.savefig(os.path.join(result_dir, 'model_loss.png'))
	plt.close()


def save_history(history, result_dir):
	### FUNCTION: creates a .txt file with the output from training
	# history: a Kears history obeject
	# result_dir: where to store the file

	loss = history.history['loss']
	acc = history.history['acc']
	val_loss = history.history['val_loss']
	val_acc = history.history['val_acc']
	nb_epoch = len(val_acc)

	with open(os.path.join(result_dir, 'result.txt'), 'w') as fp:
		fp.write('epoch\tloss\tacc\tval_loss\tval_acc\n')
		for i in range(nb_epoch):
			fp.write('{}\t{}\t{}\t{}\t{}\n'.format(i, loss[i], acc[i], val_loss[i], val_acc[i]))


def data_generator(data, labels, batch_size, imdims, numAugs = 0, phase='train', cropem=False, maskout=False, dataroot=None):

	assert phase in ['train', 'test', 'val'], "generator 'phase' must be one of 'train', 'test' or 'val'"
	assert dataroot is not None, "dataroot cannot be None"

	if phase == 'train':
		np.random.seed()
		rndidx 		= range(len(data))
		rndidx 		= np.random.choice(rndidx, len(rndidx), replace=False)
		data 		= data[rndidx]
		labels 		= labels[rndidx]

	batch 		= int(batch_size/float(numAugs)) if numAugs else batch_size
	batch_idxs 	= int(np.ceil(len(data) / float(batch)))

	while True:
		for idx in range(batch_idxs):
			#print ('Generator batch: {} =  {} - {}'.format(idx, idx*int(batch),(idx+1)*int(batch)))
			batch_files 	= data[idx*int(batch):(idx+1)*int(batch)]
			batch_labels 	= labels[idx*int(batch):(idx+1)*int(batch)]
			x, y 		= get_batch(batch_files, batch_labels, numAugs, imdims[:-1], \
						dataroot= dataroot,
						cropem  = cropem,
						maskout = maskout,
						)
			yield (x.astype('float32'), y.astype('float32'))


def main():
	parser =argparse.ArgumentParser(description='CNN Training')
	parser.add_argument('--batch', 		type=int, 	default=48)
	parser.add_argument('--epoch', 		type=int, 	default=100)
	parser.add_argument('--data', 		type=str, 	default='/data2/RCA2017/', help='directory where images are stored')
	parser.add_argument('--output', 	type=str, 	required=True)
	parser.add_argument('--skip', 		type=bool, 	default=True)
	parser.add_argument('--load_weights', 	type=bool, 	default=False)
	parser.add_argument('--load_model', 	type=bool, 	default=False)
	parser.add_argument('--cropem', 	type=bool, 	default=False)
	parser.add_argument('--maskout', 	type=bool, 	default=False)
	parser.add_argument('--numaugs', 	type=int, 	default=4)
	parser.add_argument('--verbose', 	type=int, 	default=0)
	
	args = parser.parse_args()
	cropem 	= args.cropem
	maskout = args.maskout
	numAugs = args.numaugs
	verbose = args.verbose
	imdims = [128, 128, 8, 1]

	sys.stdout.write('[*]\tSplitting train/validation/test\n')
	X_train, y_train, X_val, y_val, X_test, y_test = get_split(args.data, 'bbdata.npy', 1)


	### DEFINE THE MODEL ###
	if args.load_model:
		''' Try to load the model from the weights, if not just load from the model file'''
		weight_files 	= [os.path.join(args.output,x) for x in os.listdir(args.output) if x.startswith("weights-") and x.endswith(".h5py")]
		modelPath 	= max(weight_files , key = os.path.getctime)

		try:
			model = load_model(modelPath)
		except:
			"model path does not exist starting again: {}".format(modelPath)
		
		initial_epoch 	= int(modelPath.split('improvement-')[-1].split('-')[0])

		sys.stdout.write('[*]\tModel loaded: {}\n'.format(modelPath))
		sys.stdout.flush()
	else:
		model 		= Resnet3DBuilder.build_resnet_101([128,128,8,1], 2)
		initial_epoch 	= 0
		lr=0.0001
		model.compile(
		loss='categorical_crossentropy',
		optimizer=Adam(lr=lr, decay=0.005),
		metrics=['acc']
		)

	''' SOME HYPERPARAMETERS '''
	


	if verbose == 2:
		model.summary()

	if not os.path.exists(os.path.abspath(args.output)):
		os.makedirs(os.path.abspath(args.output))

	plot_model(model, show_shapes=True, to_file=os.path.join(args.output, 'model.png'))

	logdir 			= os.path.join(args.output, 'logs')
	checkpoint_filepath 	= os.path.join(args.output, "weights-improvement-{epoch:02d}-{val_acc:.3f}.h5py")
	checkpoint 		= ModelCheckpoint(checkpoint_filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	tbCallBack 		= TensorBoard(
					log_dir 		= logdir,
					histogram_freq 		= 0,
					write_batch_performance = True,
					write_graph 		= True,
					write_images 		= True
					)

	history = model.fit_generator(
		generator 		= data_generator(X_train, y_train, args.batch, imdims, numAugs = numAugs, cropem = cropem, maskout = maskout, dataroot=args.data),
		use_multiprocessing	= True,
		max_queue_size		= 20,
		steps_per_epoch		= int(len(X_train)/args.batch)*numAugs,
		validation_data		= data_generator(X_val, y_val, args.batch, imdims, phase='val', numAugs = 0, cropem = cropem, maskout = maskout, dataroot=args.data),
		validation_steps	= int(len(X_val)/args.batch),
		epochs			= args.epoch,
		verbose			= 1,
		callbacks		= [checkpoint, tbCallBack],
		initial_epoch 		= initial_epoch
		)

	model.save(os.path.join(args.output, 'model.h5py'))

	###### GET THE BEST VALIDATION RESULTS, LOAD THOSE WEIGHTS AND DO THE FINAL TEST ##############
	weight_files 	= [os.path.join(args.output,x) for x in os.listdir(args.output) if x.startswith("weights-") and x.endswith(".h5py")]
	newest 		= max(weight_files , key = os.path.getctime)
	model.load_weights(newest)
	sys.stdout.write('[*]\tWeights loaded: {}\n'.format(newest))
	sys.stdout.flush()

	loss, acc = model.evaluate_generator(
		generator 	= data_generator(X_test, y_test, args.batch, imdims, phase='test',numAugs = 0, cropem = cropem, maskout = maskout, dataroot=args.data),
		steps 		= int(len(X_test)/args.batch),
		max_queue_size	= 20
		)

	print('Best Test loss:', loss)
	print('Best Test acc:', acc)

	save_history(history, args.output)	

	model_json = model.to_json()
	if not os.path.isdir(args.output):
		os.makedirs(args.output)
	with open(os.path.join(args.output, 'model.json'), 'w') as json_file:
		json_file.write(model_json)


if __name__ == '__main__':
	main()
