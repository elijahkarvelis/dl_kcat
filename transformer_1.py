"""
AUTHOR: E Karvelis (karvels2@mit.edu)
DATE:   April 25, 2023

PURPOSE: 
Predict TIS-calculated k_cat values for different enzyme mutants as a 
function of their active site dynamics leading up to attempted turnover
events. Observations of the dynamics leading to, and sometimes including,
attempted turnover are referred to as trajectories or pathways. The 
goal of this script is to train a model that predicts the k_cat for a 
given mutant based on a sampled set of its trajectories.

METHOD:
This script implements a standard transformer encoder, specialized 
for handling multivariate time series data, upstream of a head that 
predicts k_cat from the pooled encodings across a set of input 
trajectories. 

(set of trajectories) >> transformer encoder >> pooling >> prediction head >> k_cat


EXECUTION:
python transformer_1.py config.txt

where the config.txt file specifies all settings. See below for info on each 
setting. An example file might read:

data_file = 'data/tptrue_gsfalse_o-0dot4_0dot8_s1_2_3_4_5_r1_2_t-110_0_sub500_numNone.470000-111-70memnpy'
meta_file = 'data/tptrue_gsfalse_o-0dot4_0dot8_s1_2_3_4_5_r1_2_t-110_0_sub500_numNone.metadata'
loc = '/data/karvelis03/dl_kcat/'
split_by_variant = True
path_set_size = 10
random_seed = 333
lr = 0.001
epochs = 100


DEPENDENCIES:
This script sources the object for structuring metadata, which is defined in 
./scripts/prep_data.py. In practice, this script imports everything from prep_data.py.


INPUT:
data_file --        Name of the data file, which stores a memory-mapped numpy array of
			        the form saved by prep_data.py. This array has form 
			        [pathways, timesteps, features]. (str)
meta_file --        Name of the metadata file, which stores the metadata describing 
			        each pathway entry (along axis 0) of the data_file. The meta_file
			        stores a saved, pickled python object generated by prep_data.py when 
			        saving the corresponding data_file (meta_file will have the same name
			        as its data_file, but the suffix of the data_file is replaced with 
			        '.metadata'). This object is an instance of Recurrent_data as defined in
			        scripts/prep_data.py. It contains information associated with each 
			        pathway along axis 0 of the data_file array: the variant, k_cat, 
			        error in k_cat, pathway type (i.e., order), ensemble seed number, 
			        ensemble statistical replicate number, time steps (along axis 1). (str)
loc --              The pathway to the location of the dl_kcat repo. This script has some 
                    dependencies, all of which it can source form the repo's ./scripts
                    and ./input subfolders. (str)
cv_folds --         The number of cross validation folds to use. Default, 5. int
split_by_variant -- Whether to split the cross validation folds by enzyme variants. 
                    That is, if True, then all the pathway data from a given variant
                    will be in only the train or test set (never both) for a given
                    cross validation fold. Each variant is held out in the test set
                    for only one of the folds. If False, then the pathway data from 
                    a given variant can be distributed between both the train and 
                    test sets for every fold. Generally, we set this arument to True,
                    because we are  interested in how the model will generalize to 
                    new variants whose data on which it hasn't been trained. (bool)
path_set_size --    The number of pathways to include per 'observation' of an enzyme
					variant. That is, the model will predict k_cat from a set of 
					path_set_size number of pathways from a given variant. The model
					architecture will be designed so that it can make this prediction
					for any variable number of pathways, but for training purposes, 
					we select the number of pathways so that the training data can 
					be appropriately packed into 'observations,' where each observation 
					is a small set of pathways. Defaults to 10. int
batch_size --       Size of training batches. Each batch will include batch size number 
                    of variant observations, where each observation is comprised of 
                    path_set_size number of pathways. Default, 32. int
random_seed --      Number with which to intialize the random number generator. int
lr --               Learning rate for the optimizer (Adam optimizer). Default, 1e-3. (float)
epochs --           Number of training epochs. Default, 100. (int)

"""

""" ============================================== """
###                     Packages                   ###
""" ============================================== """
import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
import math






""" ============================================== """
###          Helper functions and classes (put in another script one day)          ###
""" ============================================== """
class PathDataset():
	# Stores the memory-mapped numpy array and corresponding
	# Recurrent_data object with associated metadata

	def __init__(self, data_file, meta_file=None, path_set_size=10):
		self.data_file = data_file
		self.path_set_size = path_set_size
		if meta_file != None:
			self.meta_file = meta_file
		else:
			suffix = data_file.split('num')[-1].split('.')[-1]
			self.meta_file = data_file.replace(suffix, 'metadata')


		self.data = np.memmap(self.data_file, dtype='float32', mode='r', shape=self.get_data_shape(self.data_file))
		self.meta = pickle.load(open(self.meta_file, 'rb'))
		self.obs = None


	def info(self):
		print ("self.data_file -- name of the file the PathDataset is sourcing a memory-mapped numpy array from")
		print ("self.meta_file -- name of the file the PathDataset is sourcing the meta data from (Recurrent_data object)")
		print ("self.data -- the loaded memory-mapped numpy array")
		print ("self.meta -- the loaded Recurrent_data object with metadata")
		pirnt ("self.obs -- instance of Observations object. Stores indexes for each 'observation' of a mutant")


	@staticmethod
	def get_data_shape(data_file):
		suffix = data_file.split('num')[-1].split('.')[-1]
		shape = np.array(suffix.split('memnpy')[0].split('-'), dtype=int)
		shape = tuple(shape)
		return shape

	def make_observations(self):
		# Populates self.obs with an instance of the Observations class
		# 
		# This creates small groups of pathways, where each pathway
		# in a given group is from the same enzyme variant. Such
		# a group of pathways is called an 'observation' of a variant.
		# 
		# Observations are organized in an instance of the Observations 
		# class stored in self.obs.
		# Each observation has one element in the array self.obs.obs,
		# and that element is an array listing the indexes (along axis 0)
		# of self.data for the pathways belonging to that observation.
		# Also included as part of the self.obs instance are the 
		# self.obs.variant, self.obs.kcat, and self.obs.kcat_sem attibutes, 
		# which are arrays that respectively list the variant, kcat, and 
		# kcat SEM associated with each observation in self.obs.obs. Note 
		# that these attributes of self.obs are redundant in that self 
		# already has attributes containing this kind of information. The 
		# copying of variant and kcat-related metrics to self.obs is for 
		# convenience. If this causes issues later on (memory, general 
		# performance), consider doing away with the redundancy.

		self.obs = self.Observations(self)


	class Observations():
		# Populates an array called self.obs, listing the indexes (along axis 0)
		# of PathDataset.data for the pathways belonging to each observation.
		# Also creates self.variant, self.kcat, and self.kcat_sem attibutes, 
		# which are arrays that respectively list the variant, kcat, and 
		# kcat SEM associated with each observation (each row) in self.obs.
		def __init__(self, PathDataset):
			self.obs = []
			self.variant = []
			self.kcat = []
			self.kcat_sem = []

			for var in np.unique(PathDataset.meta.variant):

				var_paths = np.where(np.array(PathDataset.meta.variant) == var)[0]

				# get kcat-related metadata
				kcat = PathDataset.meta.kcat[var_paths[0]]
				kcat_sem = PathDataset.meta.kcat_sem[var_paths[0]]

				# group variant's paths into a set of 'observations'
				n_obs = int(np.floor(var_paths.shape[0] / PathDataset.path_set_size))
				var_obs = np.random.choice(var_paths, size=(n_obs,PathDataset.path_set_size), replace=False)

				# append observations to list
				self.obs.append(var_obs)

				# add metadata
				self.variant     += [var]     *var_obs.shape[0]
				self.kcat        += [kcat]    *var_obs.shape[0]
				self.kcat_sem    += [kcat_sem]*var_obs.shape[0]

			# convert all data to single numpy arrays
			self.variant = np.array(self.variant)
			self.kcat = np.array(self.kcat)
			self.kcat_sem = np.array(self.kcat_sem)
			self.obs = np.vstack(self.obs)


class NormalScaler():
	# Similar to sklearn's StandardScaler() in that it scales
	# features to have mean=0 and variance=1 by applying 
	# z = (x-u)/s where z is the udpated feature value, x 
	# the original value, u the mean, and s the standard deviation.
	#
	# Given 3D data of form [pathways, timesteps, features], each 
	# feature's u and s are calculated across all timesteps across
	# all paths. So, there are data.shape[2] number of s and u 
	# in total
	# 
	# call .fit() to fit on training data, then
	# call .transform() to scale both training
	# and testing data as needed
	# 
	# Example:
	# scaler = NormalScaler()
	# scaler.fit(data.data)
	# x = data.data[0:3,:,:]
	# x = scaler.transform(x)

	def __init__(self):
		self.avg = None
		self.std = None

	def fit(self, x):
		self.avg = np.mean(x, axis=(0,1))
		self.std = np.std(x, axis=(0,1))

	def transform(self, x):
		if not isinstance(self.avg, np.ndarray):
			raise ValueError('NormalScaler instance must first be fit to data before transforming data')

		x = (x - self.avg) / self.std

		return x


class MinMaxScaler():
	# Similar to sklearn's MinMaxScaler() in that it scales
	# features to have range from 0 to 1 by applying 
	# x_scaled = (x - min(x))/(max(x) - min(x)) where x_scaled
	# is the udpated feature value, x  the original value
	#
	# Given 3D data of form [pathways, timesteps, features], each 
	# feature's min and max are taken across all timesteps across
	# all paths. So, there are data.shape[2] number of min and max 
	# in total
	# 
	# call .fit() to fit on training data, then
	# call .transform() to scale both training
	# and testing data as needed

	def __init__(self):
		self.min = None
		self.max = None

	def fit(self, x):
		self.min = np.min(np.min(x, axis=1), axis=0)
		self.max = np.max(np.max(x, axis=1), axis=0)

	def transform(self, x):
		if not isinstance(self.min, np.ndarray):
			raise ValueError('MinMaxScaler instance must first be fit to data before transforming data')

		x = (x - self.min) / (self.max - self.min)

		return x 


class DataScaler():
	# A class that can be passed as the transform argument when 
	# making an instance of PathTorchDataset, enabling the 
	# scaling, or normalization, of data as it is called by 
	# the PyTorch Dataloader.
	# INPUT:
	# scaler -- a MinMaxScaler or NormalScaler instance that has
	#           been fit to some data, which you want to apply
	# OUTPUT -- when __call__(sample) executes, this will apply
	#           the .transform() function of scaler to the data
	#           in sample['paths'], i.e. the pathway data

	def __init__(self, scaler):
		self.scaler = scaler

	def __call__(self, sample):
		paths = scaler.transform(sample['paths'])
		return {'paths': paths, 'kcat': sample['kcat']}


class PathTorchDataset(Dataset):
	# Defines a customized Dataset class for use with 
	# PyTorch based on the standard PyTorch Dataset class
	

	def __init__(self, pathdataset, elligible_idxs=None, transform=None):
		# pathdataset -- an instance of the PathDataset object, with the 
		#                .obs attribute populated (PathDataset)
		# elligible_idxs -- the indexes along the attributes of
		#                   pathdataset.obs that are elligible for 
		#                   selection when loading data. This variable
		#                   is meant to pass the indexes of training or 
		#                   testing data. Default, None, in which case
		#                   all indexes are considered elligible. 
		#    				(numpy array, None).
		# transform -- transform to apply to samples (function, optional)

		self.pathdataset = pathdataset
		self.transform = transform
		if elligible_idxs is None:
			self.elligible_idxs = np.arange(self.pathdataset.obs.obs.shape[0])
		else:
			self.elligible_idxs = elligible_idxs

	def __len__(self):
		return (self.elligible_idxs.shape[0])

	def __getitem__(self, idx):

		# Convert idx to the index along self.path.dataset.obs
		# entries that is elligible for selection
		selected_idx = self.elligible_idxs[idx]

		# Collect paths' data
		path_idxs = self.pathdataset.obs.obs[selected_idx]
		paths = self.pathdataset.data[path_idxs,:,:]

		# Collect kcat value
		kcat = self.pathdataset.obs.kcat[selected_idx]
		# take log bc kcat values span several orders of magnitude
		log_kcat = np.float32(np.log10(kcat))
		sample = {'paths': torch.from_numpy(paths).to(device),
				  'kcat': log_kcat}

		if self.transform:
			sample = self.transform(sample)

		return sample


class PositionalEncoding(nn.Module):
	# Implementation taken from PyTorch documentation at:
	# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
	# INPUT:
	# d_model -- the dimension of the transformer encoder layers, this
	#            is the number of features in the input data after being 
	#            passed through the encoder layer. int
	# dropout -- fraction of neurons subjected to dropout. Default, 0.1.
	#			 float
	# max_len -- the dimensionality of the positional encoding array. 
	#            i.e., the array to be added to input is of shape 
	#            [max_len, 1, d_model], but only the first "input_size"
	#            rows will be added to input data. Here, "input_size"
	#            is the number of time points in the input data

    def __init__(self, d_model: int, dropout: float=0.1, max_len: int=500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        # INPUT:
        # x -- Tensor of form [seq_len, batch_size, embedding_dim], so 
        #      the data in x need to first be permuted to shift the batch
        #      dimension from axis 0 to axis 1 like so: x = x.permute(1, 0, 2)
        
        ### x = x + self.pe[:x.size(0)]
        
        """
		Can we do the below implementation instead? And require the the 
		second-to-last axis corresponds to the time points?
        """
        # print (f'x.shape: 		{x.shape}')
        # print (f'self.pe.shape: {self.pe.shape}')
        # print (f'self.pe[0:x.size(-2),:].shape: {self.pe[0:x.size(-2),:].shape}\n')
        x = x + self.pe[0:x.size(-2),:]

        return self.dropout(x)


class TransformerModel(nn.Module):
	# Encodes sets of multivariate time series with a transformer, 
	# pools the encodings, then uses the pooled encoding to predict
	# log(kcat) with an MLP prediction head

	def __init__(self, 
				 input_size: int,
				 input_length: int,
				 d_model: int=512,
				 max_input_length: int=500,
				 dropout_pos_encoder: float=0.1,
				 n_head: int=8,
				 d_tran_ffn: int=2048,
				 dropout_tran_encoder: float=0.2,
				 n_tran_layers: int=4):
		# INPUT:
		# input_size -- number of features in input. For example, 1 if univariate or 
		# 			    or 70 if using 70 structural features. int
		# d_model -- the dimensions of the transformer encoder layers in the 
		#            transformer encoder. All sublayers in the model will produce
		#            outputs with this dimension. int
		# input_length -- the length of each time series in time points. int
		# max_input_length -- the max length of the input sequence that is being fed to 
		#                 	  the model. This must be at least as large as the number of 
		#                 	  time points (i.e., input_length >= input_data.shape[-2]).
		#                     It is only used in defining the positional encoding.
		#                 	  Default, 500. int
		# dropout_pos_encoder -- dropout for the positional encoding step. Default, 0.1.
		#                        float
		# n_head -- the number of attention heads (parallel attention layers) in 
		#           each transformer encoder layer. Default, 8. int.
		# d_tran_ffn -- number of neurons in the linear feedforward layer of the transformer
		#               encoder layers. Default, 2048. int
		# dropout_tran_ecoder -- dropout for the transformer encoder layers. Default, 0.2.
		#                        float
		# n_tran_layers -- Number of stacked transformer encoder layers in the transformer
		#                  encoder. Default, 4. int
		# 
		#

		super().__init__()

		self.input_size = input_size
		self.input_length = input_length
		self.model_type = 'PredictiveTransformerEncoder'

		# Linear layer for encoding raw input
		self.input_encoder = nn.Linear(in_features=input_size, 
									   out_features=d_model)

		# Positional encoder
		self.pos_encoder = PositionalEncoding(d_model, dropout_pos_encoder, input_length)

		# Transformer encoder
		encoder_layers = nn.TransformerEncoderLayer(d_model, n_head, 
												   dim_feedforward=d_tran_ffn,
												   dropout=dropout_tran_encoder,
												   batch_first=True)
		self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_tran_layers)

		# Prediction head (MLP layer)
		self.mlp_head = nn.Linear(in_features=input_length*d_model,
								  out_features=1)


	def forward(self, src: Tensor, src_mask: Tensor=None) -> Tensor:
		# INPUT
		# src -- the input data of shape [n_paths, timesteps, features] where
		#        n_paths is the number of pathways per observation (usually path_set_size),
		#        timesteps is the number of time steps in the series, and 
		#        features is the number of features (e.g., the 70 structural features).
		#        Tensor
		# src_mask -- the mask for the src sequence to prevent the model using 
		#             specific data point. For now, this should always be set 
		#             to None, because it isn't relevant to our application.
		#  			  Default, None. None or Tensor
		# OUTPUT
		# Returns a predicted value for log10(kcat)

		# Encode input
		# print ('Initial encoding')
		# print (f'src.shape: {src.shape}')
		src = self.input_encoder(src) * math.sqrt(self.input_size)
		# print (f'src.shape: {src.shape}\n\n')

		# Add positional encoding
		# print ('Positional encoding')
		# print (f'src.shape: {src.shape}')
		src = self.pos_encoder(src)
		# print (f'src.shape: {src.shape}\n\n')

		# Pass through transformer encoder
		# print ('Transformer encoding')
		# print (f'src.shape: {src.shape}')
		# start_enc = time.time()
		# enc1 = self.transformer_encoder(src)   ### doesn't work for 4D data
		# print (f'enc runtime: {time.time() - start_enc}')
		# print (f'enc1.shape: {enc1.shape}')

		"""	Rough, brute force way to handle 4D data:
			Just pass each batch one at a time, where each
			batch consists of path_set_size number of pathways.
			The build in transformer layers in PyTorch accept the 
			3D data of a single batch (but not 4D multiple batches).
			It's like you trick it into treating the different pathways
			as different, independent batches at this stage. We want
			independence of pathways at this stage, because their order 
			shouldn't matter, and a downstream pooling operation (or 
			other trainable operations we can think about later) will
			handle the simultaneous consideration of all pathways and 
			how they relate to one another 

			NOTE: the current brute force implementation actually works
			      quite well and is just as fast for processesing something
			      like [32, 10, 111, 512] as processing [320, 111, 512]

		"""
		# start_enc = time.time()
		# enc1 = torch.zeros(src.shape)
		# for i in range(src.shape[0]):
		# 	enc1[i,:,:,:] = self.transformer_encoder(src[i,:,:,:])
		# print (f'enc runtime: {time.time() - start_enc}')
		# print (f'enc1.shape: {enc1.shape}\n\n')


		"""
		Alternative method that involves re-shaping.
		However, I'm not certain the reshaping methods
		preserve the original order and specific structure
		of the data -- need to check that.

		However, in practice the above for-loop option
		seems to run faster, or at least it for sure isn't slower
		"""
		# convert src from 4D to 3D by stacking the first axis
		start_enc = time.time()
		orig_shape = src.shape
		src = src.view(-1, orig_shape[-2], orig_shape[-1]) # double check that this is equivalent to vstacking
		enc1 = self.transformer_encoder(src)
		# convert enc1 back to 4D from 3D; this recovers separate batches along first axis
		enc1 = enc1.view(-1, orig_shape[-3], orig_shape[-2], orig_shape[-1]) # double check that this is the inverse of vstacking and recovers the batches
		# print (f'enc runtime: {time.time() - start_enc}')
		# print (f'enc1.shape: {enc1.shape}\n\n')


		# Take average across all paths in each observation (experiment with inclusion of other moments and/or max pooling)
		# print ('Averaging over paths')
		# print (f'enc1.shape: {enc1.shape}')
		enc = torch.mean(enc1, 1)
		# print (f'enc.shape: {enc.shape}\n\n')


		# Flatten each observation's averaged time series
		# print ('Flattening avg time series')
		# print (f'enc.shape: {enc.shape}')
		enc = torch.flatten(enc, start_dim=1)
		# print (f'enc.shape: {enc.shape}\n\n')

		# Prediction head
		# print ('MLP prediction head')
		# print (f'enc.shape: {enc.shape}')
		out = self.mlp_head(enc)
		# print (f'out.shape: {out.shape}')
		# print (out, '\n\n')



		return out


def train(model, dataloader) -> None:
	# Training function. Call this once for every epoch to run
	# through the data in dataloader and update the model parameters
	# INPUT:
	# model -- an instance of a PyTorch nn.Module
	# dataloader -- PyTorch DataLoader to stream training data


		model.train()
		total_loss = 0.0
		start_time = time.time()
		log_interval = 25 # print info every log_interval number of batches

		for batch_idx, batch in enumerate(dataloader):

			# forward pass
			output = model(batch['paths'])

			# print (type(output), output, output.dtype)
			# print (type(batch['kcat'].view(-1,1)), batch['kcat'].view(-1,1), batch['kcat'].view(-1,1).dtype)

			loss = loss_fn(output, batch['kcat'].view(-1,1))

			# backward pass
			optimizer.zero_grad()
			loss.backward()
			# torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) # uncomment to help prevent gradients from exploding
			# more info: https://pytorch.org/tutorials/beginner/transformer_tutorial.html

			# update weights
			optimizer.step()

			# update loss 
			total_loss += loss.item()


			if (batch_idx+1) % log_interval == 0:
				# print progress and info
				time_per_batch = (time.time() - start_time) / log_interval
				current_loss = total_loss / log_interval
				root_loss = math.sqrt(current_loss)

				print (f'epoch {epoch} | CV fold {cv_fold}/{cv_folds} | {batch_idx+1:d}/{len(dataloader):d} batches | '
					   f'lr {lr:0.3e} | seconds/batch {time_per_batch:.3f} | '
					   f'loss {current_loss:.3e} | sqrt(loss) {root_loss:.3e}')
				output_text.write(f'epoch {epoch} | CV fold {cv_fold}/{cv_folds} | {batch_idx+1:d}/{len(dataloader):d} batches | '
					   f'lr {lr:0.3e} | seconds/batch {time_per_batch:.3f} | '
					   f'loss {current_loss:.3e} | sqrt(loss) {root_loss:.3e}\n')
				output_text.flush()

				# reset loss and timer for next round of log_interval number of batches
				total_loss = 0
				start_time = time.time()


def evaluate(model, dataloader) -> float:
	# Model evaluation function. Typically, this is to be
	# executed at the end of every epoch to report the 
	# model performance on held out data
	# INPUT:
	# model -- an instance of a PyTorch nn.Module
	# dataloader -- PyTorch DataLoader to stream validation
	#               or testing data
	# RETURNS:
	# A dictionary with 'total loss' and 'avg loss' items,
	# where the 'avg loss' is the MSE, and the total loss 
	# is the sum of the SE over all observations in dataloader

		model.eval()
		total_loss = 0.0
		total_obs = 0
		with torch.no_grad():
			for batch_idx, batch in enumerate(dataloader):
				n_obs = batch['paths'].size(0)

				output = model(batch['paths'])
				loss = loss_fn(output, batch['kcat'].view(-1,1))
				total_loss += loss.item() * n_obs
				total_obs += n_obs

		avg_loss = total_loss / total_obs # this is equivalent to MSE

		return {'total loss': total_loss, 'avg loss': avg_loss}




""" ============================================== """
###                     Main code                  ###
""" ============================================== """
if __name__ == '__main__':


	""" #=============================================					
	        Load settings and set up environment
	""" #=============================================
	import sys
	import time

	# read in settings to overwrite defaults
	split_by_variant = True
	random_seed = 333
	path_set_size = 10
	batch_size = 32
	cv_folds = 5
	lr = 1e-3
	epochs = 3
	output_text = './transformer_1_output.txt'
	with open(sys.argv[1], 'r') as f:
		settings = f.read()
		exec(settings)

	# import dependencies from prep_data.py
	sys.path.append(f'{loc}/scripts/')
	from prep_data import *

	# initialize random number generator
	np.random.seed(random_seed)

	# open a file for writing progress
	output_text = open(output_text, 'w')

	# Device configuration
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print ('DEVICE: ', device)


	""" #=============================================
	                       Load data
	""" #=============================================
	data = PathDataset(data_file, meta_file, path_set_size=path_set_size)
	# group pathways into small sets (from same variant)
	data.make_observations()



	""" #=============================================
                        Cross-validation loop
	""" #=============================================
	# define CV splitter
	cv = GroupKFold(n_splits=cv_folds)
	for i, (train_idx, test_idx) in enumerate(cv.split(np.arange(data.obs.obs.shape[0]), groups=data.obs.variant)):

		cv_fold = i+1

		# Define a new model
		model = TransformerModel(input_size=data.data.shape[-1],
								 input_length=data.data.shape[-2],
								 d_model=256,
								 n_head=4,
								 n_tran_layers=2,
								 d_tran_ffn=512).to(device)

		total_params = sum(p.numel() for p in model.parameters())
		trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
		print (f'Total parameters:     {total_params}')
		print (f'Trainable parameters: {trainable_params}')

		# Define loss and optimizer, CONSIDER ADDING SCHEDULER FOR LR
		loss_fn = nn.MSELoss()
		optimizer = torch.optim.Adam(model.parameters(), lr=lr)
		# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)


		# Fit the scaler to the training data
		scaler = NormalScaler()
		scaler.fit(data.data[np.concatenate(data.obs.obs[train_idx]),:,:])
		data_scaler = DataScaler(scaler)
		# print ('\n\n\n\n Finished fitting SCALER \n\n\n\n')



		# Define datasets and dataloaders for train and test sets
		train_dataset = PathTorchDataset(data, elligible_idxs=train_idx, transform=data_scaler)
		test_dataset  = PathTorchDataset(data, elligible_idxs=test_idx, transform=data_scaler)
		train_variants = np.unique(train_dataset.pathdataset.obs.variant[train_dataset.elligible_idxs])
		test_variants = np.unique(test_dataset.pathdataset.obs.variant[test_dataset.elligible_idxs])


		print ('\n\nTraining variants:')
		print (train_variants, '\n\n')
		output_text.write('\n\nTraining variants:\n')
		output_text.write(str(train_variants))
		output_text.write('\n\n\n')
		output_text.flush()


		best_val_loss = float('inf')
		best_model_file = 'best_model.pt'
		for epoch in range(1, (epochs+1)):

			start_epoch = time.time()

			# initialize data loading
			train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
			test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

			# train and test model
			train(model, train_loader)
			val_results = evaluate(model, test_loader)

			# report results
			epoch_duration = time.time() - start_epoch
			loss, mse = val_results['total loss'], val_results['avg loss']
			print ('\n\n','-'*80)
			print (f'End of epoch {epoch} | CV fold {cv_fold}/{cv_folds} | time: {epoch_duration:.1f}s | '
         		   f'valid loss (MSE) {mse:.3f} | RMSE {mse**0.5:.3f}')
			print ('\nValidation variants:')
			print (test_variants)

			output_text.write('\n\n'+'-'*80+'\n')
			output_text.write(f'End of epoch {epoch} | CV fold {cv_fold}/{cv_folds} | time {epoch_duration:.1f}s | '
         		   			  f'valid loss (MSE) {mse:.3f} | RMSE {mse**0.5:.3f}\n')
			output_text.write('\nValidation variants:\n')
			output_text.write(str(test_variants))
			output_text.write('\n')

			if mse < best_val_loss:
				print (f'\nBest loss achieved, saving model state to {best_model_file}')
				output_text.write(f'\nBest loss achieved, saving model state to {best_model_file}\n')
				best_val_loss = mse
				torch.save(model.state_dict(), best_model_file)
				# to load into model again later:
				# model.load_state_dict(torch.load(best_model_file))

			print ('-'*80, '\n\n')
			output_text.write('-'*80+'\n\n\n')
			output_text.flush()







	output_text.flush()
	output_text.close()
	print ('DONE')



	"""
	Psuedo code:
	for train, test in cv_splits():
		
		# set up a scaler fit to training data
		scaler.fit(training_data)
	
		# Set up dataloaders for train and test sets.
		# These dataloaders can apply the scaler themselves, 
		# or you can do it yourself after they provide data
		train_dataloader = DataLoader(train)
		test_dataloader = DataLoader(test)

		# Set up a fresh new instance of the model
		model = model()

		for epoch in epochs:
			for batch in train_dataloader:
				
				[transform batch with scaler, if needed]

				model.train(batch)
				model.update()

		# evaluate model on test set
		[transform test data with scaler, if needed]
		model.score(test)
		scores.append(score)


	"""






