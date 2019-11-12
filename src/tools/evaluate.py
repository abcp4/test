import numpy as np
import os
import time
import json
from sklearn.metrics import precision_score, accuracy_score, recall_score, classification_report, confusion_matrix, \
	f1_score

class Metrics():
	def __init__(self, train_acc, dev_target, dev_pred, filenames, logits, elapsed_time, n_classes):
		self.train_acc = train_acc
		self.dev_target = dev_target
		self.dev_pred = dev_pred
		self.filenames = filenames
		self.logits = [x.tolist() for x in logits]
		self.elapsed_time = elapsed_time
		self.n_classes = n_classes

	def generate(self) :
		self.mean_valid_acc = self.mean_acc()
		self.CM = confusion_matrix( self.dev_target, self.dev_pred, labels=[n for n in range(self.n_classes)] )
		self.f1 = f1_score( self.dev_target, self.dev_pred, average = None, labels=[n for n in range(self.n_classes)] )
		self.f1_weighted = f1_score( self.dev_target, self.dev_pred, average = 'weighted', labels=[n for n in range(self.n_classes)] )
		self.recall = recall_score( self.dev_target, self.dev_pred, average = None, labels=[n for n in range(self.n_classes)] )
		self.precision = precision_score( self.dev_target, self.dev_pred, average = None, labels=[n for n in range(self.n_classes)] )
		return [self.train_acc, self.mean_valid_acc, self.recall, self.f1, self.precision, self.f1_weighted,
				self.dev_target, self.dev_pred, self.filenames, self.logits, self.elapsed_time]

	def mean_acc(self):
		c = 0
		for j in range( len( self.dev_pred ) ) :
			if self.dev_pred[j] == self.dev_target[j] :
				c += 1
		return c / ( len( self.dev_pred ) + 1e-10)

	def report(self):
		print(self.CM)
		print(classification_report( self.dev_target, self.dev_pred ))
		print('---------------------------------------------------------')
		print('F1 score per class (good,bad,ugly): ', self.f1)
		print('Mean F1 score: ', np.mean( self.f1 ))
		print('F1 weighted: ', self.f1_weighted)
		print('---------------------------------------------------------')


class Evaluate():
	def __init__(self, experiment_name, model_description, n_classes):
		self.experiment_name = experiment_name
		self.model_name = '_'.join(experiment_name.split('_')[:-1])
		self.model_description = model_description
		self.n_classes = n_classes
		self.path_to_results = 'src/results/'
		self.values_path = self.path_to_results + self.experiment_name + '/values.json'
		self.params_path = self.path_to_results + self.experiment_name + '/params.json'

	def split_metric(self, metric):
		metric = np.array( metric )
		return [ metric[:, position] for position in range(self.n_classes)]

	def save_metrics(self, td_results, params) :
		n_folds = len( td_results )
		factor = np.sqrt( n_folds )

		train_acc = [x[0] for x in td_results]
		dev_acc = [x[1] for x in td_results]
		class_recall = [x[2] for x in td_results]
		class_f1 = [x[3] for x in td_results]
		class_precision = [x[4] for x in td_results]
		class_f1_weighted = [x[5] for x in td_results]
		target = [[int(y) for y in x[6]] for x in td_results]
		pred = [[int(y) for y in x[7]]  for x in td_results]
		filenames = [[str(y) for y in x[8]]  for x in td_results]
		logits = [x[9]  for x in td_results] # Erro para o Block D
		elapsed_time = [x[10] for x in td_results]

		# Basic Metrics
		elapsed_time = np.array(elapsed_time)
		target = np.array(target)
		pred = np.array(pred)

		# Split F1
		f = np.array( class_f1 )
		f_good, f_bad, f_ugly = self.split_metric(f)
		mean_f1 = [np.mean( x ) for x in class_f1]
		f1_weighted = np.array( class_f1_weighted )

		# Calculate Deviance F1
		dev_dp_f1 = np.std( mean_f1 ) / factor
		dev_dp_f1_weighted = np.std( f1_weighted ) / factor
		dev_dp_good_f1 = np.std( f_good ) / factor
		dev_dp_bad_f1 = np.std( f_bad ) / factor
		dev_dp_ugly_f1 = np.std( f_ugly ) / factor

		# Split Precision
		precision = np.array( class_precision )
		precision_good, precision_bad, precision_ugly = self.split_metric(precision)

		# Calculate Deviance Precision
		dev_dp_precision_good = np.std( precision_good ) / factor
		dev_dp_precision_bad = np.std( precision_bad ) / factor
		dev_dp_precision_ugly = np.std( precision_ugly ) / factor

		# Split Recall
		recall = np.array( class_recall )
		recall_good, recall_bad, recall_ugly = self.split_metric(recall)

		# Calculate Deviance Recall
		dev_dp_recall_good = np.std( recall_good ) / factor
		dev_dp_recall_bad = np.std( recall_bad ) / factor
		dev_dp_recall_ugly = np.std( recall_ugly ) / factor

		# Calculate Accuracy
		train_mean_acc = np.mean( train_acc )
		dev_mean_acc = np.mean( dev_acc )

		# Calculate Deviance Accuracy
		train_dp_acc = np.std( train_acc ) / factor
		dev_dp_acc = np.std( dev_acc ) / factor

		dict_result = {
			# Experiment Base Metrics
			'experiment_name'     : self.experiment_name,
			'model_name'          : self.model_name,
			'model_description'   : self.model_description,
			'elapsed_time'        : elapsed_time.sum( )/60,
			'elapsed_time_list'        : elapsed_time.tolist( ),
			'timestamp'           : time.time( ),
			# Confusion Matrix
			'target'                : target.tolist( ),
			'pred'                : pred.tolist( ),
			'filenames'           : filenames,
			'logits'              : logits,
			# Accuracy
			'train_mean_acc'      : train_mean_acc,
			'dev_mean_acc'        : dev_mean_acc,
			# Deviance Accuracy
			'train_dp_acc'        : train_dp_acc,
			'dev_dp_acc'          : dev_dp_acc,
			# Accuracy List
			'train_acc_list'      : train_acc,
			'dev_acc_list'        : dev_acc,
			# F1
			'dev_mean_f1'         : np.mean( mean_f1 ),
			'f1_weighted'         : np.mean( f1_weighted ),
			'f1_good'             : f_good.mean( ),
			'f1_bad'              : f_bad.mean( ),
			'f1_ugly'             : f_ugly.mean( ),
			# Deviance F1
			'dev_dp_f1'           : dev_dp_f1,
			'dev_dp_f1_weighted'  : dev_dp_f1_weighted,
			'f1_dp_good'          : dev_dp_good_f1,
			'f1_dp_bad'           : dev_dp_bad_f1,
			'f1_dp_ugly'          : dev_dp_ugly_f1,
			# F1 List
			'f1_list'             : mean_f1,
			'f1_weighted_list'    : f1_weighted.tolist( ),
			'f1_good_list'        : f_good.tolist( ),
			'f1_bad_list'         : f_bad.tolist( ),
			'f1_ugly_list'        : f_ugly.tolist( ),
			# Precision
			'precision_good'      : precision_good.mean( ),
			'precision_bad'       : precision_bad.mean( ),
			'precision_ugly'      : precision_ugly.mean( ),
			# Deviance Precision
			'dp_precision_good'   : dev_dp_precision_good,
			'dp_precision_bad'    : dev_dp_precision_bad,
			'dp_precision_ugly'   : dev_dp_precision_ugly,
			# Precision List
			'precision_good_list' : precision_good.tolist( ),
			'precision_bad_list'  : precision_bad.tolist( ),
			'precision_ugly_list' : precision_ugly.tolist( ),
			# Recall
			'recall_good'         : recall_good.mean( ),
			'recall_bad'          : recall_bad.mean( ),
			'recall_ugly'         : recall_ugly.mean( ),
			# Deviance Recall
			'dp_recall_good'      : dev_dp_recall_good,
			'dp_recall_bad'       : dev_dp_recall_bad,
			'dp_recall_ugly'      : dev_dp_recall_ugly,
			# Recall List
			'recall_good_list'    : recall_good.tolist( ),
			'recall_bad_list'     : recall_bad.tolist( ),
			'recall_ugly_list'    : recall_ugly.tolist( )
		}
		# Save Results on two files
		# values.json is the dictionary with the metrics and params.json is the dictionary with network hyperparameters
		if not os.path.exists( os.path.dirname( self.values_path ) ) :
			os.makedirs( os.path.dirname( self.values_path ) )

		with open( self.values_path, 'w' ) as f :
			json.dump( dict_result, f, indent = 4, sort_keys = True )

		with open( self.params_path, 'w' ) as f :
			json.dump( params, f, indent = 4, sort_keys = True )

		print( 'saved at: ', self.params_path )

		# Print Relevant data
		del dict_result['target']
		del dict_result['pred']
		del dict_result['filenames']
		del dict_result['logits']
		print( json.dumps( dict_result, indent = 4, sort_keys = True ) )

