from keras.applications.inception_v3 import InceptionV3
from keras.layers import GlobalAveragePooling2D
from keras.models import Model

model_name = 'baseline_Inception_V3'
model_description = 'Inception_V3'

def make_model(output=None):
	#network configuration
	base_model = InceptionV3( weights = 'imagenet', include_top = False )
	if output == 'final':
		return model_description, [[model_name, base_model]]

	models = []
	for layer in range( 0, 11 ) :
		layer_name = 'mixed' + str( layer )
		x = base_model.get_layer( layer_name ).output
		x = GlobalAveragePooling2D( )( x )
		m = Model( inputs = base_model.input, outputs = x )
		models.append( ['%s_%s' % (model_name, layer_name), m] )

	return model_description, models
