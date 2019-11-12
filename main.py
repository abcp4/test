import argparse
import os
from src.tools.params import dotdict
from sklearn.model_selection import ParameterGrid
from src.tools.CrossValidation import CrossValidation
import json

class Main:
	def __init__(self):
		# Initial variables
		self.config_path = 'src/config/'
		self.dataset_path = '../dataset/'
		self.result_path = ''

	def load_config(self, config):
		print('Loading parameters from %s%s.json' % (self.config_path, config) )
		json_path = os.path.join( self.config_path, config ) + '.json'
		with open( json_path ) as f :
			JSON = json.load( f )
		return dotdict( JSON )

	def filter_grid_none(self, dict):
		grid = {}
		for option in dict :
			if options[option] != None :
				grid[option] = options[option]
		return grid

	def get_grid_search(self, options):
		# Filter Options that are Null
		grid = self.filter_grid_none( options )
		# Search for parameters
		if grid != {}: grid = ParameterGrid( grid )
		return grid

	def run_workflow(self, model, config):
		if model != None:
			return
		# Loading Config
		params = self.load_config(config)
		# Run model
		options['model'] = model
		grid_search = self.get_grid_search(options)
		print('Available options for GridSearch: ', list(grid_search))

		# For each option in grid search
		for search in grid_search:
			# Change parameters
			for key in search.keys():
				params[str(key)] = search[key]

if __name__ == '__main__':
	# Load Configs from terminal
	parser = argparse.ArgumentParser(
		prog = 'Seismic Classification Program',
	    description = "Seismograms Classification Noise between Good(G), Bad(B) and Ugly(U) Categories" )

	parser.add_argument('-baseline', type = str, help="Run Baseline Model", nargs = '+', choices=['vgg16','vgg19','inceptionV3'] )
	parser.add_argument('-baseline_config', default='baseline', help="Load Baseline Hyperparameters", nargs = '?')
	parser.add_argument('-miniception', type = str, help="Run Miniception Model", nargs = '+', choices=['A','B','C','D','D_ordinal'] )
	parser.add_argument('-holdout', type = str, help = "Run Miniception Holdout", nargs = '+', choices=['A','B','C','D','D_ordinal'] )
	parser.add_argument('-config', default='miniception_R17', help="Load Miniception Hyperparameters")
	parser.add_argument('-gpu_config', default='gpu', help="Load GPU Configuration", nargs='?')
	parser.add_argument( '-gpu', type = str, help = "GPU Selection", nargs = '?' )
	parser.add_argument('-predict', help = "Evaluate Model", nargs = '+' )
	parser.add_argument('-fit_predict', help = "Retrain and Evaluate Model", nargs = '+' )
	parser.add_argument('-fit_predict_threshold', help = "Evaluate Model", nargs = '+' )

	parser.add_argument('-slice',  type = float, help = "Subset Slice from Dataset", nargs = '+' )
	parser.add_argument('-num_epochs',  type = int, help = "Number of Epochs", nargs = '+' )
	parser.add_argument('-batch',  type = int, help = "Batch Size", nargs = '+' )
	parser.add_argument('-lr',  type = float, help = "Learning Rate", nargs = '+' )
	parser.add_argument('-alpha',  type = int, help = "Alpha Search", nargs = '+' )
	parser.add_argument('-beta',  type = int, help = "Beta Search", nargs = '+' )
	parser.add_argument('-width',  type = int, help = "Width Search", nargs = '+' )
	parser.add_argument('-height',  type = int, help = "Height Search", nargs = '+' )
	parser.add_argument('-n_blocks',  type = int, help = "Number of Blocks Search", nargs = '+' )
	parser.add_argument('-n_folds',  type = int, help = "Number of Cross-Validation Folds", nargs = '+' )
	parser.add_argument('-aug',  type = bool, default =[False], help = "Augmentation. True or False", nargs = '+' )
	parser.add_argument('-build_aug',  type = bool, default =[False], help = "Build Aug images in folder", nargs = '+' )
	parser.add_argument('-aug_images', type = int, help = "Number of Augmented Images", nargs = '+' )
	parser.add_argument( '-segy', type = str, help = "SEGY Reader. True or False", nargs = '+' )
	parser.add_argument( '-labels_path', type = str, default = '', help = "SEGY Labels. True or False", nargs = '+' )

	# Loading Arguments
	args = parser.parse_args()
	m = Main( )
	# Performing Grid Search
	# define parameters that may vary
	options = {'slice': args.slice,'num_epochs': args.num_epochs, 'batch': args.batch,
	           'alpha': args.alpha, 'beta': args.beta,'n_blocks': args.n_blocks, 'n_folds': args.n_folds,
			   'width': args.width,'height': args.height, 'augmented':args.aug,
	           'build_aug': args.build_aug, 'aug_images':args.aug_images, 'learning_rate':args.lr,
	           }
	gpu_config = m.load_config( args.gpu_config )
	if args.gpu: gpu_config.gpu = args.gpu # Set GPU

	for experiment in [(args.baseline,args.baseline_config, 'baseline'),
					   (args.miniception,args.config,'miniception'),
	                   (args.holdout, args.config,'holdout'),
	                   (args.predict, args.config,'predict'),
	                   (args.fit_predict, args.config,'fit_predict'),
	                   (args.fit_predict_threshold, args.config,'fit_predict_threshold')]:

		if experiment[0] != None:
			# Loading Configs
			params = m.load_config( experiment[1] )
			# Define model
			options['model'] = experiment[0]
			grid_search = m.get_grid_search( options )
			print('Available options for GridSearch %s: ' % experiment[0], list( grid_search ))
			# For each option in grid search
			for search in grid_search :
				# Change parameters
				for key in search.keys( ) :
					params[str( key )] = search[key]
				# Run Model Search
				print('Runnig Search for hyperparameters: ', params)
				c = CrossValidation( gpu_config, params )
				params['config_name'] = experiment[1]
				params['mode'] = experiment[2]
				if experiment[2] == 'baseline' :
					c.baseline( )
				elif experiment[2] == 'miniception' :
					c.workflow_default( )
				elif experiment[2] == 'holdout' :
					c.workflow_test( )
				elif experiment[2] == 'predict' :
					c.predict( )
				elif experiment[2] == 'fit_predict' :
					c.fit_predict( )
				elif experiment[2] == 'fit_predict_threshold' :
					c.fit_predict_threshold( )
