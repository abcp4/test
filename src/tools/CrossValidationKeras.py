import os
import tensorflow as tf
from src.tools import DatasetBuilder
from src.models.MiniceptionD import MiniceptionD
from sklearn.model_selection import StratifiedKFold
import time


class CrossValidation():
	def __init__(self, session_config):
		if session_config.gpu != None:
			os.environ["CUDA_VISIBLE_DEVICES"] = session_config.gpu
		os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
		os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
		os.environ['TF_KERAS'] = '1'
		tf.logging.set_verbosity( tf.compat.v1.logging.ERROR )

		self.path =  str( time.time( ) )
		self.seed = 12345
		self.create_session(session_config)
		self.strategy = tf.distribute.MirroredStrategy()
		self.checkpoint_dir = './training_checkpoints/' + self.path
		self.checkpoint_prefix = os.path.join( self.checkpoint_dir, "ckpt_{epoch}" )
		self.save_path = 'saved_model/' + self.path + '/'

	def create_session(self, session_config) :
		config = tf.ConfigProto( )
		config.gpu_options.allow_growth = session_config.allow_growth
		tf.keras.backend.set_session( tf.Session( config = config ) )

	def load_data_seismogram(self, params, filename, data_folder, build=True):
		build_aug = False
		if build: build_aug = params.build_aug
		data_train = DatasetBuilder( filename, data_folder,
		                             beta = params.beta, width = params.width, height = params.height,
		                             list_augment = params.list_augment, aug_images = params.aug_images,
		                             build_aug = build_aug,
		                             rgb = params.rgb, normalize = params.normalize )
		filename, _class = data_train.load_file( )
		filename, _class = data_train.split( filename, _class, slice = params.slice )
		filename, _class = data_train.process( filename, _class, augmented = params.augmented )
		return filename, _class

	def process_path(self, filename, label) :
		img = tf.read_file(	 filename )
		img = tf.image.decode_image( img )
		img.set_shape([None, None, 3])
		img = tf.image.rgb_to_grayscale( img )
		img = tf.cast( 2 * (img / 255 - 0.5), tf.float32 )
		label = tf.cast( label, tf.int64 )
		return img, label

	def make_dataset(self, filename, labels, batch_size, training=True) :
		dataset = tf.data.Dataset.from_tensor_slices((filename, labels))
		if training:
			dataset = dataset.shuffle( 10000, seed=self.seed )
		dataset = dataset.map( self.process_path )
		dataset = dataset.batch( batch_size )
		dataset = dataset.prefetch( 10 )
		return dataset

	def build_model(self, alpha, n_blocks, params):
		model = MiniceptionD( alpha = alpha, n_blocks = n_blocks )
		model.build( (None, params.width, params.height, params.n_channels) )
		model.compile(
			loss = 'sparse_categorical_crossentropy',
			optimizer = tf.keras.optimizers.Adam( 0.00001),
			metrics = ['accuracy'] )
		return model

	def decay(self, epoch) :
		initial_lrate = 0.00001
		return initial_lrate * (0.99 ** epoch)

	def callbacks(self, model):
		return [
			tf.keras.callbacks.TensorBoard( log_dir = './logs' ),
			tf.keras.callbacks.ModelCheckpoint( filepath = self.checkpoint_prefix,
			                                    save_weights_only = True ),
			#tf.keras.callbacks.LearningRateScheduler( self.decay ),
			PrintLR( model )
		]

	def create_folds(self, n_folds, X, Y) :
		kf = StratifiedKFold( n_splits = n_folds, shuffle = True, random_state = self.seed )
		return kf.split( X, Y )

	def workflow_default(self, params):
		global_batch_size = (params.batch * self.strategy.num_replicas_in_sync)
		print ('Number of devices: {}'.format(self.strategy.num_replicas_in_sync))
		# Carrega os Dados
		X, Y = self.load_data_seismogram(params, params.label_filename, params.data_folder, build=True)
		print(X.shape, Y.shape)

		# Cria os folds
		folds = self.create_folds(params.n_folds, X, Y)

		for fold_index, (train_folds, test_folds) in enumerate(folds):
			print('Processing Fold : ', fold_index + 1)
			fold_start = time.time()
			# Constroi o Modelo
			model = self.build_model(alpha=params.alpha, n_blocks=params.n_blocks, params= params)
			# model.build((None, params.width, params.height, params.n_channels))

			# Define o Callback
			callbacks = self.callbacks(model)

			# Cria o Dataset
			train_dataset = self.make_dataset(X[train_folds], Y[train_folds], global_batch_size)
			test_dataset = self.make_dataset(X[test_folds], Y[test_folds], global_batch_size, training=False)

			# Treina o modelo
			model.fit(train_dataset, epochs=params.num_epochs, callbacks=callbacks)

			# Avalia o modelo
			model.load_weights(tf.train.latest_checkpoint(self.checkpoint_dir))
			eval_loss, eval_acc = model.evaluate(test_dataset)
			print ('Eval loss: {}, Eval Accuracy: {}'.format(eval_loss, eval_acc))

	def workflow_test(self, params):
		global_batch_size = (params.batch * self.strategy.num_replicas_in_sync)
		print ('Number of devices: {}'.format( self.strategy.num_replicas_in_sync ))
		# Carrega os Dados
		x_train, y_train = self.load_data_seismogram( params, params.label_filename, params.data_folder, build = True )
		x_test, y_test = self.load_data_seismogram( params, params.label_test_filename, params.data_test_folder )
		print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

		# Cria o Dataset
		train_dataset = self.make_dataset( x_train, y_train, global_batch_size )
		test_dataset = self.make_dataset( x_test, y_test, global_batch_size, training = False )

		# Constroi o Modelo
		model = self.build_model( alpha = params.alpha, n_blocks = params.n_blocks)
		model.build((None, params.width, params.height, params.n_channels))
		print(model.summary())

		# Define o Callback
		callbacks = self.callbacks( model )

		# Treina o modelo
		model.fit( train_dataset, epochs = params.num_epochs, callbacks = callbacks )

		# Salva o modelo
		model.save( f'{self.save_path}.h5' )

		# Avalia o modelo
		model.load_weights( tf.train.latest_checkpoint( self.checkpoint_dir ) )
		eval_loss, eval_acc = model.evaluate( test_dataset )
		print ('Eval loss: {}, Eval Accuracy: {}'.format( eval_loss, eval_acc ))

	def restore_checkpoint(self, params,
						   checkpoint_path="./training_checkpoints/1569871183.195606/ckpt_"):
		# Create test
		global_batch_size = (params.batch * self.strategy.num_replicas_in_sync)
		x_test, y_test = self.load_data_seismogram( params, params.label_test_filename, params.data_test_folder )
		test_dataset = self.make_dataset( x_test, y_test, global_batch_size, training = False )

		# Create Model
		model = self.build_model(alpha=params.alpha, n_blocks=params.n_blocks)
		model.build((None, params.width, params.height, params.n_channels))

		# Load Weigths
		model.load_weights(checkpoint_path)

		# Treinar em um novo conjunto

		# Evaluate
		eval_loss, eval_acc = model.evaluate( test_dataset )
		print ('Eval loss: {}, Eval Accuracy: {}'.format( eval_loss, eval_acc ))

	def restore(self, params, model_path=''):
		# Load Model
		model = tf.keras.models.load_model( model_path )

		# Create test
		global_batch_size = (params.batch * self.strategy.num_replicas_in_sync)
		x_test, y_test = self.load_data_seismogram( params, params.label_test_filename, params.data_test_folder )
		test_dataset = self.make_dataset( x_test, y_test, global_batch_size, training = False )

		# Evaluate
		eval_loss, eval_acc = model.evaluate( test_dataset )
		print ('Eval loss: {}, Eval Accuracy: {}'.format( eval_loss, eval_acc ))
		print(model.predict_classes(test_dataset))
		# Salvar as predicoes em uma pasta

# Callback for printing the LR at the end of each epoch.
class PrintLR(tf.keras.callbacks.Callback):
	def __init__(self, model):
		self.model = model
	def on_epoch_end(self, epoch, logs=None):
		print ('\nLearning rate for epoch {} is {}'.format(
		    epoch + 1, tf.keras.backend.get_value(self.model.optimizer.lr)))

