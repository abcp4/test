import time
from imp import reload

import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, train_test_split, KFold
from tqdm import tqdm

from src.tools.Baseline import Baseline
from src.tools.DatasetBuilder import DatasetBuilder
from src.tools.Session import Session
from src.tools.evaluate import Evaluate, Metrics
from src.tools.segy_reader import SegyReader


class CrossValidation( Session ) :
	def __init__(self, config, params, checkpoint_path='checkpoints/'):
		super().__init__(config)
		self.params = params
		self.sess = self.create_session()
		self.checkpoint_path = checkpoint_path

		# Cria o Model
		self.model, is_baseline = self.get_model( self.params.model )
		if is_baseline:
			baseline = Baseline( )
			self.model_description, self.layers = baseline.build( self.model )
		else:
			self.build_model( model = self.model, width = self.params.width, height = self.params.height,
		                  n_classes = self.params.n_classes,
		                  alpha = self.params.alpha, n_blocks = self.params.n_blocks,
		                  n_channels = self.params.n_channels )

		if params.segy:
			# Tem que poder fazer o pedido sem a labels path
			test_reader = SegyReader(
				path = params.segy,
				labels_path = "../classification/test_png.txt",
				batch_size = 1
			)
			print( 'Test SEGY ', test_reader )

			self.X_train, self.Y_train = [], []
			self.X_test, self.Y_test = test_reader
		else:
			# Carrega os arquivos de treinamento
			data_train = DatasetBuilder( self.params.label_filename, self.params.data_folder,
			                             beta = self.params.beta, width = self.params.width, height = self.params.height,
			                             list_augment = self.params.list_augment, aug_images = self.params.aug_images,
			                             build_aug = self.params.build_aug,
			                             rgb = self.params.rgb, normalize = self.params.normalize, add_pos = False )

			filename, _class = data_train.load_file( )
			filename, _class = data_train.split( filename, _class, slice = self.params.slice )
			self.X_train, self.Y_train = data_train.process( filename, _class, augmented = self.params.augmented )

			# Carrega os arquivos de teste
			
			data_test = DatasetBuilder( self.params.label_test_filename, self.params.data_test_folder,
			                            beta = self.params.beta, width = self.params.width, height = self.params.height,
			                            rgb = self.params.rgb, normalize = self.params.normalize, add_pos = False )
			filename_test, _class_test = data_test.load_file( label = 'Test' )
			

			filename_test, _class_test = data_test.split( filename_test, _class_test, slice = 1 )
			self.X_test, self.Y_test = data_test.process( filename_test, _class_test )


		# Concatena ambos
		self.X = np.concatenate( [self.X_train, self.X_test] )
		self.Y = np.concatenate( [self.Y_train, self.Y_test] )
		self.indices = np.arange( self.X.shape[0] )

	def workflow_test(self) :
		"""Holdout"""

		train_folds = self.indices[:self.X_train.shape[0]]
		test_folds = self.indices[self.X_train.shape[0] :]

		start = time.time( )
		td_results = self.train_schedule( X = self.X, Y = self.Y, train_folds = train_folds, dev_folds = [],
										  test_folds = test_folds,
										  max_epochs = self.params.num_epochs, early_stop_epochs = 0,
										  batch_size = self.params.batch,
										  lr = self.params.learning_rate )
		td_results.append( time.time( ) - start )
		Evaluate( self.model_name, self.model_description, self.params.n_classes ).save_metrics( [td_results], self.params )

	def workflow_default(self) :

		td_results = self.run( X = self.X_train, Y = self.Y_train, n_folds = self.params.n_folds, epochs = self.params.num_epochs,
							   batch_size = self.params.batch, learning_rate = self.params.learning_rate,
							   early_stop_epochs = self.params.early_stop_epochs,
							   early_stop = self.params.early_stop )
		Evaluate( self.model_name, self.model_description, self.params.n_classes ).save_metrics( td_results, self.params )

	def baseline(self) :
		# Carrega os arquivos de treinamento/teste
		data = DatasetBuilder( self.params.label_filename, self.params.data_folder, rgb = self.params.rgb,
							   beta = self.params.beta, width = self.params.width, height = self.params.height,
							   normalize = self.params.normalize, add_pos = False )
		filename, labels = data.load_file( )
		
		filename, _class = data.split( filename, labels, slice = self.params.slice )

		images = data.load_images( filename, width = self.params.width, height = self.params.height,
								   augmented = self.params.augmented )
		labels = data.load_labels( _class )

		for layer in self.layers[2 :3] :
			self.model_name, _ = layer
			values = self.baseline.extract_values( X = images, layer = layer )
			td_results = self.run( X = images, Y = labels, n_folds = self.params.n_folds, baseline = True,
								   dev_pred = values )
			Evaluate( self.model_name, self.model_description, self.params.n_classes ).save_metrics( td_results, self.params )

	def fit_predict_threshold(self, threshold=[0,0,0]):
		test_folds = np.arange(len(self.X_test))

		td_results = self.train_schedule( X = self.X_test, Y = self.Y_test,
										  train_folds = [], dev_folds = [], test_folds = test_folds,
		                                  max_epochs = 0, early_stop_epochs = 0,
		                                  batch_size = self.params.batch,
		                                  lr = self.params.learning_rate, checkpoint = self.params.checkpoint,
		                                  optimizer = threshold )

		print('Best F1 in test: ', td_results[5])
		# Evaluate( self.model_name, self.model_description, self.params.n_classes ).save_metrics( td_results, self.params )

	def fit_predict_threshold_2(self):
		# Get a portion to adjust threshold
		split = 100
		indices = np.arange(len(self.X_test))
		train_folds, test_folds = train_test_split(indices, train_size=split, stratify=self.Y_test)

		# Threshold_search
		best_optimizer = [0, 0, 0]
		best_f1 = 0

		for dx in range(-10, 10, 1):
			for dy in range(-10, 10, 1):
				optimizer = [0, dx/10000, dy/10000]
				td_results = self.train_schedule(X=self.X_test, Y=self.Y_test,
												 train_folds=[], dev_folds=[], test_folds=train_folds,
												 max_epochs=0, early_stop_epochs=0,
												 batch_size=self.params.batch,
												 lr=self.params.learning_rate, checkpoint=checkpoint, optimizer=optimizer)
				f1 = td_results[5]
				print(f1, optimizer)
				if f1 >= best_f1:
					best_f1 = f1
					best_optimizer = optimizer

		td_results = self.train_schedule( X = self.X_test, Y = self.Y_test,
										  train_folds = [], dev_folds = [], test_folds = test_folds,
		                                  max_epochs = 0, early_stop_epochs = 0,
		                                  batch_size = self.params.batch,
		                                  lr = self.params.learning_rate, checkpoint = self.params.checkpoint,
		                                  optimizer = best_optimizer )

		print('Best F1 in test: ', td_results[5])
		# Evaluate( self.model_name, self.model_description, self.params.n_classes ).save_metrics( td_results, self.params )

	def predict(self):
		test_folds = np.arange( len( self.X_test ) )

		td_results = self.train_schedule( X = self.X_test, Y = self.Y_test,
										  train_folds = [], dev_folds = [], test_folds = test_folds,
		                                  max_epochs = 0, early_stop_epochs = 0,
		                                  batch_size = self.params.batch,
		                                  lr = self.params.learning_rate, checkpoint = self.params.checkpoint )

		Evaluate( self.model_name, self.model_description, self.params.n_classes ).save_metrics( [td_results], self.params )

	def fit_predict(self):

		# Get a portion to adjust threshold
		indices = np.arange( len( self.X_test ) )
		max_sample = int(len(indices)/self.params.n_folds)

		results = []
		kf = StratifiedKFold( n_splits = self.params.n_folds, shuffle = True, random_state = self.seed )
		for fold_index, (test_folds, train_folds) in enumerate( kf.split( indices, self.Y_test ) ) :
			print( 'FOLD ', fold_index + 1 )
			fold_results = []
			for subsample in range( 1, max_sample + 1 ) :
				if subsample == max_sample :
					train, dev = train_folds, []
				else :
					train, dev = train_test_split( train_folds, train_size = subsample, shuffle = False, random_state = self.seed )
				print(len(train))

				td_results = self.train_schedule( X = self.X_test, Y = self.Y_test, train_folds = train,
				                                  dev_folds = [],
				                                  test_folds = test_folds,
				                                  max_epochs = self.params.num_epochs, early_stop_epochs = 0,
				                                  batch_size = self.params.batch,
				                                  lr = self.params.learning_rate, checkpoint = self.params.checkpoint,
								  fold_index = fold_index)
				self.sess.close()
				self.sess = self.create_session()

				fold_results.append( td_results )
			results.append( fold_results )

		for index in range(max_sample):
			group = [x[index] for x in results]

			Evaluate( self.model_name, self.model_description, self.params.n_classes ).save_metrics( group, self.params )

	def get_model(self, model) :
		baseline = False
		if model == 'vgg16' :
			import src.models.baseline_vgg16 as model
			baseline = True
		elif model == 'vgg19' :
			import src.models.baseline_vgg19 as model
			baseline = True
		elif model == 'inceptionV3' :
			import src.models.baseline_inception as model
			baseline = True
		elif model == 'A' :
			import src.models.miniception_A as model
		elif model == 'B' :
			import src.models.miniception_B as model
		elif model == 'C' :
			import src.models.miniception_C as model
		elif model == 'D' :
			import src.models.miniception_D as model
		elif model == 'D_ordinal' :
			import src.models.miniception_D_ordinal as model
		return model, baseline

	def build_model(self, model, width, height, n_classes, alpha, n_blocks, n_channels) :
		# Import model from miniception
		reload( model )
		import src.models.architecture_manager
		reload( src.models.architecture_manager )

		# get model from input placeholder variable and number of classes
		self.x, self.y, self.lr_placeholder, self.output_logits, self.pred, self.model_description, self.tscores = model.make_model(
			n_classes, alpha = alpha, width = width, height = height, seed = self.seed, n_blocks = n_blocks,
			n_channels = n_channels
		)
		# get loss tensor, accuracy tensor, and optimizer function
		if self.tscores != None:
			# IF model uses t scores, for example Miniception D_ordinal, then use it as loss function
			self.loss, self.accuracy, self.optimizer = model.make_model_loss( self.y, self.lr_placeholder, self.tscores )
		else:
			self.loss, self.accuracy, self.optimizer = model.make_model_loss( self.y, self.lr_placeholder, self.output_logits )
		# define model name
		self.model_name = model.model_name + '_' + str( time.time( ) )

	def create_folds_2(self, n_folds, X, Y) :
		split = []
		path_to_folds = '/content/dataset/train_test'
		import os
		
		folds = os.listdir(path_to_folds)
		print(folds)
		
		folds=[]
		for i in range(10):
		    folds.append(str(i))
		print(folds)
		for fold_index in folds:
			import pandas as pd
			print(f'{path_to_folds}/{fold_index}')
			train = pd.read_csv(f'{path_to_folds}/{fold_index}/train.txt', names=['filename', 'class'])
			test = pd.read_csv(f'{path_to_folds}/{fold_index}/test.txt', names=['filename', 'class'])
			

			
			cond_train = np.isin([file.split('/')[-1] for file in X], train.filename.values)
			cond_test = np.isin([file.split('/')[-1] for file in X], test.filename.values)
			
			print(cond_train)
			print(cond_test)
			
			index_train, index_test = np.where(cond_train == True)[0], np.where(cond_test == True)[0]
			print(index_train)
			print(index_test)
			a=2/0
			print('>>>>>>>>', index_train.shape, index_test.shape, self.params.augmented)
			if self.params.aug_images:
				arqs = os.listdir(f'{path_to_folds}/{fold_index}')
				augment = [x for x in arqs if x.endswith('.png')]
				augment = augment[:self.params.aug_images]
				augment = [f'{path_to_folds}/{fold_index}/' + img for img in augment]
				X = np.concatenate([X, augment])
				index_augment = np.where(np.isin(X, augment))[0]
				index_train = np.concatenate((index_train, index_augment))

				Y = np.concatenate([Y, [2] * len(augment)])
			print('>>>>>>>>', index_train.shape, index_test.shape)
			split.append([ index_train , index_test])
		return split, X, Y

	def create_folds(self, n_folds, X, Y) :
		kf = StratifiedKFold( n_splits = n_folds, shuffle = True, random_state = self.seed )
		return kf.split( X, Y )

	def run(self, X, Y, n_folds, epochs=40, batch_size=64, learning_rate=0.0001, early_stop_epochs=0, early_stop=False,
			baseline=False, dev_pred=None) :
		assert n_folds > 1, 'Tamanho de folds deve ser maior que 1'
		
		folds, X, Y = self.create_folds_2( n_folds, X, Y )
		td_results = []
		for fold_index, (train_folds, test_folds) in enumerate( folds ) :
			print('Model : ', self.model_name)
			print('Processing Fold : ', fold_index + 1)
			if baseline :
				result = self.baseline.predict( train_folds, test_folds, dev_pred, Y )
			else :
				if early_stop :
					train_folds, dev_folds = train_test_split( train_folds, train_size = 0.9,
															   stratify = Y[train_folds] )
				else :
					dev_folds = []
				result = self.train_schedule( X, Y,
											  train_folds, dev_folds, test_folds, max_epochs = epochs,
											  early_stop_epochs = early_stop_epochs,
											  batch_size = batch_size, lr = learning_rate )
			td_results.append( result )
		return td_results

	def elapsed_time(self, start, end) :
		fold_time = end - start
		print('fold processing time: ', fold_time, ' s')
		return fold_time

	def process_path(self, filename, label,name) :
		img = tf.read_file( filename, name='read_file' )
		img = tf.image.decode_image( img, name='decode', channels=3, expand_animations=False )
		#img.set_shape( [None, None, 3] )
		if self.params.n_channels == 1:
			img = tf.image.rgb_to_grayscale( img )
		img = tf.cast( 2 * (img / 255 - 0.5), tf.float32 )
		label = tf.cast( label, tf.int64 )
		return img, label,name

	def make_dataset(self, filename, labels, batch_size, training=True) :
		dataset = tf.data.Dataset.from_tensor_slices( (filename, labels,filename) )
		if training:
			dataset = dataset.shuffle( 10000, seed = self.seed )
		dataset = dataset.map( self.process_path, num_parallel_calls=4 )
		dataset = dataset.batch( batch_size )
		dataset = dataset.prefetch( 10 )
		return dataset

	def make_iterator(self, training_dataset) :
		dataset_iterator = training_dataset.make_initializable_iterator( )
		next_element = dataset_iterator.get_next( )
		return dataset_iterator, next_element


	def train_schedule(self, X, Y, train_folds, dev_folds, test_folds, max_epochs, early_stop_epochs, batch_size, lr,
					    checkpoint=None, optimizer=None,fold_index=0) :
		fold_start = time.time( )
		checkpoint_file = self.checkpoint_path + self.model_name + '/model'

		features_placeholder = tf.placeholder( tf.string, shape = (None,) )
		labels_placeholder = tf.placeholder( tf.int64, shape = (None,) )

		dataset_train = self.make_dataset( features_placeholder, labels_placeholder, batch_size )
		dataset_val = self.make_dataset( features_placeholder, labels_placeholder, batch_size, training = False )
		dataset_test = self.make_dataset( features_placeholder, labels_placeholder, batch_size, training = False )

		training_init_op, next_train = self.make_iterator( dataset_train )
		validation_init_op, next_val = self.make_iterator( dataset_val )
		test_init_op, next_test = self.make_iterator( dataset_test )

		self.saver = tf.train.Saver( )
		self.sess.run( tf.global_variables_initializer( ) )
		merged = tf.summary.merge_all( )

		if checkpoint != None:
			checkpoint = tf.train.latest_checkpoint( f'{self.checkpoint_path}{checkpoint}/' )
			self.saver.restore( self.sess, checkpoint )
			if optimizer != None:
				threshold = tf.get_default_graph().get_tensor_by_name("Variable:0")
				value = self.sess.run( threshold )

				value += optimizer
				threshold = tf.assign(threshold, value)
				k = self.sess.run( threshold )
				print(k)

		if len( dev_folds ) == 0 : print('no early stopping, save last model at epoch ', max_epochs)
		# Number of training iterations in each epoch
		best_dev_acc = 0
		death_counter = 0
		pbar = tqdm( range( 1, max_epochs + 1 ) )
		for epoch in pbar :
			# Print the status message.
			pbar.set_description( "\r- Training epoch: {}".format( epoch - 1 ) )
			lr *= 0.999
			# Treinamento
			self.sess.run( training_init_op.initializer,
				feed_dict = { features_placeholder : X[train_folds], labels_placeholder : Y[train_folds] } )
			trainbar = tqdm( range( 0, len( train_folds ), batch_size ) )
			for iteration in trainbar :
				trainbar.set_description( "\r- Train Batch" )

				x_batch, y_batch,z_batch = self.sess.run( next_train )
				

				feed_dict_batch = { self.x : x_batch, self.y : y_batch, self.lr_placeholder : lr }
				self.sess.run( self.optimizer, feed_dict = feed_dict_batch )
				
				

			trainbar.close( )

			# Avaliacao
			if len( dev_folds ) > 0 :
				self.sess.run( validation_init_op.initializer,
							   feed_dict = { features_placeholder : X[dev_folds], labels_placeholder : Y[dev_folds] } )
				mean_valid_acc = 0
				valid_count = 0
				validationbar = tqdm( range( 0, len( dev_folds ), batch_size ) )
				
				for iteration in validationbar :
					validationbar.set_description( "\r- Validation Batch" )

					x_batch, y_batch,z_batch = self.sess.run( next_val )
					

					feed_dict_batch = { self.x : x_batch, self.y : y_batch }
					loss_valid, acc_valid = self.sess.run( [self.loss, self.accuracy], feed_dict = feed_dict_batch )
					

					n = len( y_batch )
					valid_count += n
					mean_valid_acc += acc_valid * n
				
				validationbar.close( )

				mean_valid_acc /= valid_count
				
				if mean_valid_acc > best_dev_acc :
					best_dev_acc = mean_valid_acc
					self.saver.save( self.sess, checkpoint_file )
					death_counter = 0
					
				else :
					death_counter += 1

				if death_counter >= early_stop_epochs :
					break
			

			elif epoch == max_epochs :
				print('\n Save last model \n')
				self.saver.save( self.sess, checkpoint_file )
		pbar.close( )

		# Treinamento - Acuracia
		# self.saver.restore( self.sess, checkpoint_file )
		self.sess.run( training_init_op.initializer,
					   feed_dict = { features_placeholder : X[train_folds], labels_placeholder : Y[train_folds] } )
		mean_train_acc = 0
		train_count = 0
		trainAccBar = tqdm( range( 0, len( train_folds ), batch_size ) )
		for iteration in trainAccBar :
			trainAccBar.set_description( "\r- Train Acc" )

			x_batch, y_batch,z_batch = self.sess.run( next_train )

			feed_dict_batch = { self.x : x_batch, self.y : y_batch }
			loss_train, acc_train = self.sess.run( [self.loss, self.accuracy], feed_dict = feed_dict_batch )

			n = len( y_batch )
			train_count += n
			mean_train_acc += acc_train * n
		mean_train_acc = mean_train_acc / (train_count + 1e-10)

		# Teste
		# self.saver.restore( self.sess, checkpoint_file )
		self.sess.run( test_init_op.initializer,
					   feed_dict = { features_placeholder : X[test_folds], labels_placeholder : Y[test_folds] } )
		dev_pred = []
		dev_target = []
		logits = []
		data=[[],[],[],[]]

		testbar = tqdm( range( 0, len( test_folds ), batch_size ) )
		for iteration in testbar :
			testbar.set_description( "\r- Test" )

			x_batch, y_batch,z_batch = self.sess.run( next_test )
			data[0].extend(y_batch.tolist())
			data[1].extend(z_batch.tolist())
			

			feed_dict_batch = { self.x : x_batch, self.y : y_batch }
			valid_pred, logit = self.sess.run( [self.pred, self.tscores] , feed_dict = feed_dict_batch )

			dev_pred.extend( valid_pred )
			dev_target.extend( y_batch )
			logits.extend( logit )
			data[2].extend(logit.tolist())

		testbar.close( )

		# Compute time of processing
		fold_end = time.time( )
		elapsed_time = self.elapsed_time( fold_start, fold_end )
		# Instatiate Metrics
		m = Metrics( train_acc = mean_train_acc, dev_target = dev_target, dev_pred = dev_pred,
		             filenames = X[test_folds], logits = logits, elapsed_time = elapsed_time, n_classes = self.params.n_classes )
		# Compute Metrics
		#metrics = m.generate( )
		#m.report( )
		log_score = open("log_score.txt","a")
		log_score.write('fold_index: '+str(fold_index) + "\n")	   
		log_score.write('logits: '+str(data[2]) + "\n")
		log_score.write('names: '+str(data[1]) + "\n")
		log_score.write('accuracy:'+str(mean_train_acc)+'\n')
		log_score.close()
		import pickle
		pickle.dump([data[1],data[2],mean_train_acc],open('data'+str(fold_index)+'.p','wb'))
		print("SAVED!!")

		return metrics
