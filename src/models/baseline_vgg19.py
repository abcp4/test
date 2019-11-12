from keras.applications.vgg19 import VGG19
from keras.layers import GlobalAveragePooling2D
from keras.models import Model

model_name = 'baseline_VGG_19'
model_description = 'VGG_19'

def make_model(output=None):
	#network configuration
	base_model = VGG19( weights = 'imagenet', include_top = False )
	if output == 'final':
		return model_description, [[model_name, base_model]]

	models = []
	for layer in range( 1, 6 ) :
		layer_name = 'block' + str( layer ) + '_pool'
		x = base_model.get_layer( layer_name ).output
		x = GlobalAveragePooling2D( )( x )
		m = Model( inputs = base_model.input, outputs = x )
		models.append( ['%s_%s' % (model_name, layer_name), m] )

	return model_description, models

