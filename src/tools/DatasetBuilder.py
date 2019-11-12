# coding=utf-8
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2
from PIL import Image
import numpy as np
from .utils import inverse_list
import os
from pathlib import Path

class DatasetBuilder( ) :
	def __init__(self, label_filename, data_folder, beta, width, height, list_augment=None, aug_images=100,
	             build_aug=False, rgb=False, normalize=True, add_pos=False) :
		self.label_filename = label_filename
		self.data_folder = data_folder
		self.original_folder = '/dataset/FIGURAS_ML_PUC_2019_Co/'
		self.augmented_folder = '/dataset/augmented/'
		id_to_class = ['good', 'bad', 'ugly']
		self.class_to_id = inverse_list( id_to_class )
		self.list_augment = list_augment
		self.aug_images = aug_images
		self.path_to_root = str( Path( os.path.dirname(os.path.abspath(__file__)) ).parent.parent )
		self.path_to_dataset = '/dataset/'
		self.path_to_classification = '/src/classification/'
		self.rgb = rgb
		self.normalize = normalize
		self.add_pos = add_pos
		self.gray = [128,128,128]
		#self.width, self.height = self.reshape(beta, width, height)
		if build_aug: self.generate_augmentation()

	def summary(self, y, label='') :
		print( '------------------------------------------------' )
		print( '%s Classification File Summary' % label ); print( )
		print( 'A Total of %s files.' % y.shape[0] ); print( )
		print( 'Distribution' );
		print( y.value_counts( ) ); print( )
		print( 'Proportion' );
		print( y.value_counts( ) / y.shape[0] ); print( )
		print( '------------------------------------------------' )

	def load_file(self, label = 'Original' ) :
		print( 'Loading Classification File' )
		try:
			labels = pd.read_csv( self.path_to_root + self.path_to_classification + self.label_filename, header = None )
			labels.columns = ['filename', 'class']
			self.summary( labels['class'], label = label )  # Print Sumário do Dataset
			return labels['filename'], labels['class']
		except FileNotFoundError:
			return pd.Series([]), pd.Series([])

	def split(self, x, y, slice=1.0) :
		# Se slice for igual a 1(dataset inteiro) é retornado direto
		if slice == 1 :
			return x, y
		# Se slice tem valores entre o range de Split padrão é dividido o dataset
		elif slice > 0 and slice < 1 :
			x, X_test, y, y_test = train_test_split( x, y, train_size = slice, stratify = y )
			self.summary( y, label = 'Splited' )  # Print Sumário do Dataset
			return x.reset_index(drop=True), y.reset_index(drop=True)
		# Casos onde o slice é 0, negativo, ou maior que 1
		elif slice != 1 :
			raise Exception( 'Valor de slice está incompativel. Deve ser entre 0(exclusivo) e 1(inclusive)' )

	def process(self, filename, _class, augmented=None):
		labels = _class.map( lambda x : self.class_to_id[x] )
		filename = self.path_to_root + self.path_to_dataset + self.data_folder + '/' + filename
		if augmented:
			for techinique in self.list_augment:
				aug_folder = self.path_to_root + self.augmented_folder + techinique + '/'
				folder_images = os.listdir(aug_folder)[:self.aug_images]

				aug_data = pd.read_csv(self.path_to_root + self.path_to_classification + self.label_filename, header=None)
				aug_data.columns = ['filename', 'class']
				aug_data = aug_data[aug_data['filename'].isin(folder_images)]

				aug_images = aug_folder + aug_data['filename']
				filename = filename.append( aug_images )

				aug_labels = aug_data['class'].map( lambda x : self.class_to_id[x] )
				labels = labels.append( aug_labels )
			self.summary( labels, label = 'Augmented' )
		return filename.values, labels.values

	def generate_augmentation(self):
		path_dataset = self.path_to_root + self.original_folder
		files = os.listdir( path_dataset )
		for techinique in self.list_augment:
			aug_folder = self.path_to_root + self.augmented_folder + techinique + '/'
			try: os.mkdir(aug_folder)
			except: pass
			techinique = techinique.split('_')
			print('Salvando Augmented Images da transformação ', techinique)
			techinique, parameters = techinique[0], techinique[1:]
			for index, filename in enumerate( tqdm( files[:self.aug_images] ) ) :
				if filename.endswith( '.png' ) or filename.endswith( '.jpg' ) :
					# lê a imagem
					picture = cv2.imread( path_dataset + filename )
					# Faz as principais transformações
					if techinique == 'augV' :
						picture = cv2.flip( picture, 1 )
					if techinique == 'augH' :
						picture = cv2.flip( picture, 0 )
					if techinique == 'augB' :
						picture = cv2.flip( picture, -1 )
					if techinique == 'cmp' or techinique == 'vcmp' :
						v, To = parameters
						v, To = float(v), float(To)
						newImage = picture.copy( )
						x = np.array( [float( i ) for i in range( picture.shape[0] )] )
						transform = np.ceil( np.sqrt( (x / v) ** 2 + To ** 2 ) - To )
						for column in range( picture.shape[1] ) :
							for line, transformation in enumerate( transform ) :
								newImage[line, column] = picture[int( transformation ), column]
						picture = newImage
					# Faz a imagem caber no valor especificado
					#picture = self.resize_to_fit(picture)
					# Flipa na vertical as imagens de vcmp
					if techinique == 'vcmp' :
						picture = cv2.flip( picture, 1 )
					# Adiciona Padding
					#picture = self.add_padding(picture)
					cv2.imwrite( aug_folder + filename, picture )

	def resize_to_fit(self, picture, shape_max=(200,200)):
		# resize para tamanho menor
		w = shape_max[0] / picture.shape[1]
		h = shape_max[1] / picture.shape[0]
		p = min( h, w )
		w = int( p * picture.shape[1] )
		h = int( p * picture.shape[0] )
		return cv2.resize( picture, (w, h), interpolation = cv2.INTER_AREA )

	def add_padding(self, picture, shape_max=(200,200)):
		# adiciona padding
		delta_w = shape_max[0] - picture.shape[1]
		delta_h = shape_max[1] - picture.shape[0]
		top, bottom = 0, delta_h
		left, right = 0, delta_w
		return cv2.copyMakeBorder( picture, top, bottom, left, right, cv2.BORDER_CONSTANT, value = self.gray )

	def load_labels(self, _class, augmented=None):
		labels = np.array( _class.map( lambda x : self.class_to_id[x] ) )
		if augmented:
			for techinique in self.list_augment:
				aug_folder = self.path_to_root + self.augmented_folder + techinique + '/'
				aug_images = os.listdir(aug_folder)[:self.aug_images]

				aug_labels = pd.read_csv(self.label_filename, header=None)
				aug_labels.columns = ['filename', 'class']
				aug_labels = aug_labels[aug_labels['filename'].isin(aug_images)]['class'].map( lambda x : self.class_to_id[x] )

				labels = np.concatenate([labels, aug_labels.values])
		return labels

	def load_images(self, images, width, height, augmented=False) :
		# Load Images to Array
		X = []
		pbar = tqdm( images )
		for index, image_path in enumerate(pbar):
			# Load image
			X.append( self.image_processing( image_path , self.data_folder, width, height, pbar) )
		if augmented:
			for techinique in self.list_augment :
				aug_folder = self.path_to_root+ self.augmented_folder+techinique+'/'
				aug_images = os.listdir(aug_folder)[:self.aug_images]
				aug_labels = pd.read_csv(self.label_filename, header=None)
				aug_labels.columns = ['filename', 'class']
				aug_images = aug_labels[aug_labels['filename'].isin(aug_images)]['filename'].values
				pbar2 = tqdm( aug_images )
				for index, image_path in enumerate( pbar2 ) :
					X.append( self.image_processing(image_path,
											self.augmented_folder+techinique+'/', width, height, pbar2, techinique) )
		return np.array(X)

	def image_processing(self, image_path, data_folder, width, height, pbar=None,augmentation=None) :
		if pbar:
			pbar.set_description( "\r- Processing folder {} : {:>30} ".format( data_folder , image_path ) )

		image = cv2.imread( self.path_to_root + self.path_to_dataset + data_folder + '/' + str(image_path) )
		# changing image shape
		try:
			if width != None and height != None :
				image = cv2.resize( np.array( image ), (width, height), interpolation = cv2.INTER_AREA )
		except:
			raise Exception('Não foi possível processar a imagem ',
			                self.path_to_root + self.path_to_dataset +  data_folder + '/' + str(image_path), image)

		# Convert to Pillow module
		image = Image.fromarray( image )

		# image to numpy array
		imdata = image.getdata( )
		imsize = image.size
		vdata = np.array( imdata )
		# Corrigindo o tamanho para as novas imagens g2_...
		if not self.rgb :  # len(vdata.shape) > 1:
			vdata = np.mean( vdata, axis = 1 )
		# sets all images to (-1,+1) range
		if self.normalize :
			vdata = 2 * (vdata / 255.0 - 0.5)
		# # generate positions feats
		if self.add_pos :
			dim = imsize[0]
			idx = np.arange( dim )
			idx_rel = idx / (dim - 1)
			pos_mat_arr = np.dstack( np.meshgrid( idx_rel, idx_rel ) ).reshape( dim * dim, 2 )

			vdata = np.hstack( (vdata, pos_mat_arr) )
		if self.rgb :
			vdata = vdata.reshape( (imsize[0], imsize[1], vdata.shape[1]) )
		else :
			vdata = vdata.reshape( (imsize[0], imsize[1], 1) )
		return vdata

	def translade(self, img, width) :
		rows,cols, _ = img.shape
		for linha in range( rows - 1, width - 1, -1 ) :
			for i in range( cols ) :
				img[linha][i] = img[linha - width][i]
		for linha in range( 0, width ) :
			for i in range( cols ) :
				img[linha][i] = [122, 122, 122] # GREY
		return img

	def divide_chunks(self, l, n) :
		# looping till length l
		for i in range( 0, len( l ), n ) :
			yield l[i :i + n]

	def img_augmentation(self, filenames):
		self.clean_augmentaded()
		print( 'Augmenting %s to %s' % (self.data_folder, self.augmented_folder)  )
		pbar = tqdm( filenames )
		for arq in pbar :
			if arq.endswith('jpg') or arq.endswith('png') and not arq.startswith('aug') :
				img = cv2.imread( self.data_folder + arq )

				# flip img horizontally, vertically, and both axes with flip()
				horizontal_img = cv2.flip( img, 0 )
				vertical_img = cv2.flip( img, 1 )
				both_img = cv2.flip( img, -1 )

				# writing images
				status = cv2.imwrite( self.path_to_root +self.augmented_folder + 'augH/' + arq, horizontal_img )
				cv2.imwrite( self.path_to_root + self.augmented_folder + 'augV/' + arq, vertical_img )
				cv2.imwrite( self.path_to_root + self.augmented_folder + 'augB/' + arq, both_img )

				pbar.set_description( "\r- Processing image: {:>30} Status: {} ".format( self.data_folder + arq, status ) )

	def clean_augmentaded(self) :
		for path in self.techniques:
			arqs = os.listdir( self.path_to_root + self.augmented_folder + path   )
			print( 'Cleaning ', path )
			for arq in tqdm(arqs) :
				os.remove( self.path_to_root + self.augmented_folder + path + arq  )

	def get_augmented(self, encapsulated):
		filenames = encapsulated[0]
		for technique in self.techniques:
			files = []
			labels = []
			for filename in filenames[0]:
				files.append(self.augmented_folder+ technique + filename)
			for label in filenames[1]:
				labels.append(label)
			encapsulated.append([files,labels])
		return encapsulated
