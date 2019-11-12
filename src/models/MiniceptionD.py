import tensorflow as tf
from tensorflow.keras.activations import relu, sigmoid
from tensorflow.keras.layers import Conv2D, Dense, concatenate, GlobalAveragePooling2D, Reshape, AveragePooling2D, Input, MaxPooling2D, Flatten, Layer, Permute, multiply, Reshape
from keras import backend as K

def squeeze_excite_block(input, ratio=2):
	''' Create a channel-wise squeeze-excite block
	Args:
		input: input tensor
		filters: number of output filters
	Returns: a keras tensor
	References
	-   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
	'''
	init = input
	channel_axis = 1 if K.image_data_format() == "channels_first" else -1
	filters = init.shape.as_list()[channel_axis]
	se_shape = (1, 1, filters)

	se = GlobalAveragePooling2D()(init)
	se = Reshape(se_shape)(se)
	se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
	se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

	if K.image_data_format() == 'channels_first':
		se = Permute((3, 1, 2))(se)

	x = multiply([init, se])
	return x

class SqueezeExcite(Layer):
	def __init__(self, ratio=2, name='Squeeze', **kwargs) :
		self.built = False
		self.ratio = ratio
		super(SqueezeExcite, self ).__init__( name = name, **kwargs )

	def build(self, input_shape):
		init = input_shape
		channel_axis = 1 if K.image_data_format( ) == "channels_first" else -1
		filters = init._keras_shape[channel_axis]
		se_shape = (1, 1, filters)
		self.average = GlobalAveragePooling2D( )
		self.reshpe = Reshape( se_shape )
		self.dense_1 = Dense( filters // self.ratio, activation = 'relu', kernel_initializer = 'he_normal', use_bias = False )
		self.dense_2 = Dense( filters, activation = 'sigmoid', kernel_initializer = 'he_normal', use_bias = False )

		super( SqueezeExcite, self ).build( input_shape )

	def call(self, inputs):
		se = self.average(inputs)
		se = self.reshape(se)
		se = self.dense_1(se)
		se = self.dense_2(se)
		if K.image_data_format( ) == 'channels_first' :
			se = Permute( (3, 1, 2) )( se )
		x = multiply( [inputs, se] )
		return x

# class BlockD(tf.keras.layers.Layer):
class BlockD(object):
	def __init__(self, num_maps_1, num_maps_2, name='BlockD', **kwargs):
		self.num_maps_1 = num_maps_1
		self.num_maps_2 = num_maps_2
		self.name = name
		self.built = False
		# super(BlockD, self).__init__(name, **kwargs)

	def build(self, input_shape):
		# declara todas as layers do bloco dentro de uma lista.
		if self.built:
			return
		self.conv_aux = Conv2D( filters = self.num_maps_2, kernel_size = [1, 1],
						   strides = 1, activation = tf.nn.relu, padding = 'VALID' )
		#self.squeeze = SqueezeExcite( )
		self.conv1 = Conv2D( filters = self.num_maps_1, kernel_size = [3, 3],
						strides = 1, activation = tf.nn.relu, padding = 'SAME' )
		self.conv2 = Conv2D( filters = self.num_maps_1, kernel_size = [5, 5],
						strides = 1, activation = tf.nn.relu, padding = 'SAME' )
		self.conv5 = Conv2D( filters = self.num_maps_2, kernel_size = [3, 3], strides = 1,
						activation = tf.nn.relu, padding = 'SAME' )
		# super( BlockD, self ).build( input_shape )
		self.built = True

	def __call__(self, *args, **kwargs):
		inputs = args[0]
		self.build(inputs.shape)
		se = self.conv_aux(inputs)
		se = squeeze_excite_block(se)
		conv1 = self.conv1(inputs)
		conv2 = self.conv2(inputs)
		x = self.concat = concatenate( [conv1, conv2], axis = 3 )
		conv5 = self.conv5(x)
		return conv5 * se

class MiniceptionD(tf.keras.Model):
	def __init__(self, alpha=1, n_blocks=4, name='MiniceptionD', **kwargs):
		self.alpha = alpha
		self.n_blocks = n_blocks
		super( MiniceptionD, self ).__init__(name, **kwargs)

	def build(self, input_shape) :
		self.blocks = []
		for n_block in range(self.n_blocks):
			self.blocks.append(
				BlockD(
					num_maps_1 = 2**(n_block) * self.alpha,
					num_maps_2 = 2**(n_block+1) * self.alpha
				) )
		self.pool = MaxPooling2D( pool_size = [5, 5], strides = 2, padding = 'same' )
		self.dense1 = Dense( 32 * self.alpha, activation = 'relu' )
		self.dense2 = Dense( 3, activation = 'softmax' )
		# super( MiniceptionD, self ).build( input_shape )

	def call(self, inputs):
		x = inputs
		for block in self.blocks:
			x = block(x)
			x = self.pool( x )

		x = tf.reduce_mean(x, axis=[1, 2])
		x = self.dense1( x )
		x = self.dense2( x )
		return x