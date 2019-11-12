from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import sys
import numpy as np
from src.tools.evaluate import Metrics

seed = 12345

class Baseline():
	def build(self, model, output=None) :
		model_description, models = model.make_model( output )
		return model_description, models

	def extract_values(self, X, layer) :
		self.model_name, self.base_model = layer

		num_images = len( X )
		result = [None] * num_images
		# Extract features
		for i in range( num_images ) :
			# Status-message. Note the \r which means the line should overwrite itself.
			msg = "\r- Processing image: {0:>6} / {1}".format( i + 1, num_images )

			# Print the status message.
			sys.stdout.write( msg )
			sys.stdout.flush( )

			# Process the image and store the result for later use.
			result[i] = self.predict_values( images = X[i] )

		result = np.squeeze( np.array( result ) )

		print()
		return result

	def predict_values(self, images) :
		# Transorfm images in an arrray of images
		x = image.img_to_array( images )
		# Prepare the image for the VGG model
		x = np.expand_dims( x, axis = 0 )
		x = preprocess_input( x )

		# Calculate transfer values from Keras
		values = self.base_model.predict( x )

		# Reduce to a 1-dim array.
		values = np.squeeze( values )

		return values

	def predict(self, train_folds, test_folds, dev_pred, Y) :
		print('train : %s, test : %s' % (len( train_folds ), len( test_folds )))

		X_train, Y_train = dev_pred[train_folds], Y[train_folds]
		X_test, Y_test = dev_pred[test_folds], Y[test_folds]

		class_model = SVC( C = 1, class_weight = 'balanced', verbose = 1, random_state = seed )
		class_model.fit( X_train, Y_train )
		print()
		y_pred_test = class_model.predict( X_test )
		y_pred_train = class_model.predict( X_train )

		m = Metrics( dev_target = Y_test, dev_pred = y_pred_test )
		metrics = m.generate( )
		mean_train_acc = accuracy_score( Y_train, y_pred_train )
		metrics.insert( 0, mean_train_acc )
		m.report( )

		return metrics

	def predict_kmeans(self, train_folds,dev_pred, Y):
		from sklearn.cluster import KMeans
		kmeans = KMeans( random_state = 12345 ).fit( dev_pred[train_folds] )
		print(kmeans.labels_)
		map = {0:0, 1:2, 2:0, 3:1, 4:1, 5:1, 6:0, 7:1}
		print([map[x] for x in kmeans.labels_])
		print(Y[train_folds])
		return

	def predict_euclidean(self, train_folds, test_folds, dev_pred, Y, golden=None):
		print('train : %s, test : %s' % (len( train_folds ), len( test_folds )))

		X_train, Y_train = dev_pred[train_folds], Y[train_folds]
		X_test, Y_test = dev_pred[test_folds], Y[test_folds]

		vector = self.euclidean_vector( X_train, golden )
		y_pred_train = self.soft_max( vector )
		vector = self.euclidean_vector( X_test, golden )
		y_pred_test = self.soft_max( vector )

		m = Metrics( dev_target = Y_test, dev_pred = y_pred_test )
		metrics = m.generate( )
		mean_train_acc = accuracy_score( Y_train, y_pred_train )
		metrics.insert( 0, mean_train_acc )
		m.report( )
		return metrics

	def soft_max(self, X):
		return [triple.index(min(triple)) for triple in X]

	def euclidean_vector(self, X, golden):
		arr = []
		for value in X :
			arr.append( [self.euclidean_distance( value, golden_value ) for golden_value in golden] )
		return arr

	def euclidean_distance(self, x1, x2) :
		distance = 0.0
		import math
		for i in range( len( x1 ) ) :
			distance += pow( x1[i] - x2[i], 2 )
		return math.sqrt( distance )