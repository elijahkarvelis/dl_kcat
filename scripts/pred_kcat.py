"""
AUTHOR: E Karvelis (karvels2@mit.edu)
PURPOSE:
Helper functions and classes for the dl_kcat repo
"""

# Import dependencies
import pandas as pd
import numpy as np
import pickle
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns
import os
from glob import glob
import time
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score as AUROC

# From PyTorch
import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader


"""
Helper functions
"""
def sem(a):
    # Returns standard error of the mean (SEM) of values in array a
    
    std = np.std(a, ddof=1)
    sem = std / ((a.shape[0])**0.5)
    return sem



"""
Custom classes for data piping and handling with PyTorch
"""
class PathDataset():
	# Stores the memory-mapped numpy array and corresponding
	# Recurrent_data object with associated metadata

	def __init__(self, data_file, meta_file=None, path_set_size=10, selected_variants='*', task='kcat regression'):
		# INPUT:
		# All input are as described at the top of this file

		self.data_file = data_file
		self.path_set_size = path_set_size
		self.selected_variants = selected_variants
		self.task = task

		# set self.uniform parameter:
		# whether each observation of a variant in the self.obs
		# object is comprised of pathways of the same type. If
		# True, then each observation will contain pathways that
		# are all either reactive (R) or non-reactive (NR), but
		# never both. When False (default), observations can 
		# contain a mixture of all types of pathways. (bool)
		if self.task == 'kcat regression':
			self.uniform = False
		elif self.task == 'NR/R binary classification':
			self.uniform = True
		elif self.task == 'S/F binary classification':
			self.uniform = False

		if meta_file != None:
			self.meta_file = meta_file
		else:
			suffix = data_file.split('num')[-1].split('.')[-1]
			self.meta_file = data_file.replace(suffix, 'metadata')


		self.data = np.memmap(self.data_file, dtype='float32', mode='r', shape=self.get_data_shape(self.data_file))
		self.meta = pickle.load(open(self.meta_file, 'rb'))
		self.obs = None

		if self.selected_variants == '*':
			# then use all the variants
			self.selected_variants = np.unique(self.meta.variant)

	def info(self):
		print ("self.data_file -- name of the file the PathDataset is sourcing a memory-mapped numpy array from")
		print ("self.meta_file -- name of the file the PathDataset is sourcing the meta data from (Recurrent_data object)")
		print ("self.data -- the loaded memory-mapped numpy array")
		print ("self.meta -- the loaded Recurrent_data object with metadata")
		print ("self.obs -- instance of Observations object. Stores indexes for each 'observation' of a mutant")


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
			self.order = []

			for var in np.unique(PathDataset.meta.variant):

				if var in PathDataset.selected_variants:
					
					var_paths = np.where(np.array(PathDataset.meta.variant) == var)[0]

					# get kcat-related metadata
					kcat = PathDataset.meta.kcat[var_paths[0]]
					kcat_sem = PathDataset.meta.kcat_sem[var_paths[0]]

					# group variant's paths into a set of 'observations'
					if not PathDataset.uniform:
						# then each observation may contain a mix of both R and NR pathways
						n_obs = int(np.floor(var_paths.shape[0] / PathDataset.path_set_size))
						var_obs = np.random.choice(var_paths, size=(n_obs,PathDataset.path_set_size), replace=False)
					else:
						# then each observation may only contain either R or NR pathways, not both
						var_paths_nr = var_paths[np.where(np.array(PathDataset.meta.order)[var_paths] != 0.8)]
						n_obs_nr = int(np.floor(var_paths_nr.shape[0] / PathDataset.path_set_size))
						var_obs_nr = np.random.choice(var_paths_nr, size=(n_obs_nr,PathDataset.path_set_size), replace=False)

						var_paths_r = var_paths[np.where(np.array(PathDataset.meta.order)[var_paths] == 0.8)]
						n_obs_r = int(np.floor(var_paths_r.shape[0] / PathDataset.path_set_size))
						var_obs_r = np.random.choice(var_paths_r, size=(n_obs_r,PathDataset.path_set_size), replace=False)

						var_obs = np.vstack((var_obs_nr,var_obs_r))
					
					# append observations to list
					self.obs.append(var_obs)

					# add metadata
					self.variant     += [var]     *var_obs.shape[0]
					self.kcat        += [kcat]    *var_obs.shape[0]
					self.kcat_sem    += [kcat_sem]*var_obs.shape[0]
					self.order.append(np.array(PathDataset.meta.order)[var_obs])

			# convert all data to single numpy arrays
			self.variant = np.array(self.variant)
			self.kcat = np.array(self.kcat)
			self.kcat_sem = np.array(self.kcat_sem)
			self.obs = np.vstack(self.obs)
			self.order = np.vstack(self.order)

class PathTorchDataset(Dataset):
	# Defines a customized Dataset class for use with 
	# PyTorch based on the standard PyTorch Dataset class
	

	def __init__(self, pathdataset, elligible_idxs=None, transform=None, control_model=False):
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
		# control_model -- if True, then the targets (labels), kcat, 
		# 				   will be randomly shuffled. (bool)

		self.pathdataset = pathdataset
		self.transform = transform
		self.control_model = control_model
		if elligible_idxs is None:
			self.elligible_idxs = np.arange(self.pathdataset.obs.obs.shape[0])
		else:
			self.elligible_idxs = elligible_idxs
		
		if self.control_model:
			self.elligible_idxs_shuffled = self.elligible_idxs.copy()
			np.random.shuffle(self.elligible_idxs_shuffled)

	def __len__(self):
		return (self.elligible_idxs.shape[0])

	def __getitem__(self, idx):

		# Convert idx to the index along self.path.dataset.obs
		# entries that is elligible for selection
		selected_idx = self.elligible_idxs[idx]

		# Collect paths' data
		path_idxs = self.pathdataset.obs.obs[selected_idx]
		paths = self.pathdataset.data[path_idxs,:,:]

		if self.control_model:
			# Update selected index to sample scrambled (random) target labels
			selected_idx = self.elligible_idxs_shuffled[idx]

		# Collect kcat value
		kcat = self.pathdataset.obs.kcat[selected_idx]
		log_kcat = np.float32(np.log10(kcat))
		kcat_sem = np.float32(self.pathdataset.obs.kcat_sem[selected_idx])

		# report whether value kcat is fast or slow, wrt WT
		kcat_class = 1 if log_kcat > -16.02 else 0
		kcat_class = np.float32(kcat_class)

		# Collect paths' order information
		order = self.pathdataset.obs.order[selected_idx]
		# we'll report order as the average across all the paths'
		# orders in the observation. If pathdataset.uniform, then
		# the average is exactly the order of all the paths in the
		# observation. Otherwise when pathdataset.uniform is False, 
		# the average will range from the lowest NR order to the 
		# R order depending on how many paths in the observation 
		# were R vs. NR
		order = np.mean(order)
		# encode the order parameter, if needed
		if self.pathdataset.task == 'NR/R binary classification': 
			order = 1 if order==0.8 else 0
		order = np.float32(order)

		sample = {'paths': paths,
				  'kcat': log_kcat,
				  'kcat sem': kcat_sem,
				  'order': order,
				  'kcat_class': kcat_class}

		if self.transform:
			sample = self.transform(sample)
		else:
			try:
				for i in sample:
					sample[i] = torch.from_numpy(sample[i]).to(device)
			except:
				# executes when PathTorchDataset is imported and executed by external programs
				for i in sample:
					sample[i] = torch.from_numpy(np.array(sample[i])).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

		return sample

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

	def __init__(self, scaler, stoch_labels=False):
		self.scaler = scaler
		self.stoch_labels = stoch_labels

	@staticmethod
	def sample_kcat(mean, sem):
		# Samples a value from a normal distribution with mean = mean
		# and standard deviation = sem. Given that kcat must be > 0,
		# in the event that the sampled kcat < 0 (occurs 3-6% of time), 
		# new values are drawn until the sampled kcat is > 0
		kcat = -1
		while kcat <= 0:
			kcat = np.random.normal(loc=mean, scale=sem, size=1)[0]
		kcat = np.float32(np.log10(kcat))
		return kcat 

	def __call__(self, sample):

		paths = self.scaler.transform(sample['paths'])

		if self.stoch_labels:
			# sample kcat value from normal dist with mean = average 
			# kcat and std = SEM of kcat. Note: sample['kcat'] is in 
			# log units, so we take to the power of 10
			kcat = self.sample_kcat(10**sample['kcat'], sample['kcat sem'])
		else:
			kcat = sample['kcat']
		
		try:
			t_sample = {'paths': torch.from_numpy(paths).to(device),
						'kcat': torch.from_numpy(np.array(kcat)).to(device),
						'order': torch.from_numpy(np.array(sample['order'])).to(device),
						'kcat_class': torch.from_numpy(np.array(sample['kcat_class'])).to(device)}
		except:
			# executes when PathTorchDataset is imported and executed by external programs
			t_sample = {'paths': torch.from_numpy(paths).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')),
						'kcat': torch.from_numpy(np.array(kcat)).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')),
						'order': torch.from_numpy(np.array(sample['order'])).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')),
						'kcat_class': torch.from_numpy(np.array(sample['kcat_class'])).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))}
			
		return t_sample



"""
Custom PyTorch classes of specific machine learning models
"""
class TransformerModel(nn.Module):
	# Encodes sets of multivariate time series with a transformer, 
	# pools the encodings, then uses the pooled encoding to predict
	# log(kcat) with an MLP prediction head

	def __init__(self, 
				 input_size: int,
				 input_length: int,
				 d_input_enc: int=128,
				 d_model: int=256,
				 max_input_length: int=500,
				 dropout_pos_encoder: float=0.1,
				 n_head: int=4,
				 d_tran_ffn: int=1024,
				 dropout_tran_encoder: float=0.2,
				 n_tran_layers: int=2,
				 d_mlp_head: int=128,
				 dropout_mlp_head: float=0.2,
				 task: str='kcat regression'):
		# INPUT:
		# input_size -- number of features in input. For example, 1 if univariate or 
		# 			    or 70 if using 70 structural features. int
		# input_length -- the length of each time series in time points. int
		# d_input_enc -- hidden layer size for the (middle layer of the) 2 layer input encoder
		# d_model -- the dimensions of the transformer encoder layers in the 
		#            transformer encoder. All sublayers in the model will produce
		#            outputs with this dimension. int
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
		# d_mlp_head -- hidden layer size for the (middle layer) of the two layer MLP
		#               regression head
		# dropout_mlp_head -- dropout rate applied in between the two layers in the 
		#                     MLP regression head
		# task -- the type of learning task
		# 
		#

		super().__init__()

		self.input_size = input_size
		self.input_length = input_length
		self.model_type = 'PredictiveTransformerEncoder'
		self.task = task

		# Linear layer for encoding raw input
		self.input_encoder = nn.Sequential(nn.Linear(in_features=input_size, out_features=d_input_enc),
										   nn.ReLU(), #nn.Dropout(0.1) ???
							 			   nn.Linear(in_features=d_input_enc, out_features=d_model))

		# Positional encoder
		self.pos_encoder = PositionalEncoding(d_model, dropout_pos_encoder, input_length)

		# Transformer encoder
		encoder_layers = nn.TransformerEncoderLayer(d_model, n_head, 
												   dim_feedforward=d_tran_ffn,
												   dropout=dropout_tran_encoder,
												   batch_first=True)
		self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_tran_layers)

		# Prediction head (MLP layer)
		self.mlp_head = nn.Sequential(nn.Linear(in_features=d_model, out_features=d_mlp_head),
									  nn.ReLU(),
									  nn.Dropout(dropout_mlp_head),
									  nn.Linear(in_features=d_mlp_head, out_features=1))

		self.sigmoid = nn.Sigmoid() # used when self.task == 'NR/R binary classification'


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
		src = self.input_encoder(src) * np.sqrt(self.input_size)
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
		orig_shape = src.shape
		# print ('Transformer')
		# print (f'src.shape: {src.shape}')
		src = src.view(-1, orig_shape[-2], orig_shape[-1]) # double check that this is equivalent to vstacking
		# print (f'src.shape: {src.shape}')
		enc1 = self.transformer_encoder(src)
		# convert enc1 back to 4D from 3D; this recovers separate batches along first axis
		enc1 = enc1.view(-1, orig_shape[-3], orig_shape[-2], orig_shape[-1]) # double check that this is the inverse of vstacking and recovers the batches
		# print (f'enc1.shape: {enc1.shape}\n\n')


		# Do average pooling for each pathway across its time points
		# print (f'enc1.shape: {enc1.shape}')
		enc1 = torch.mean(enc1, 2)
		# print (f'enc1.shape: {enc1.shape}')


		# Take average across all paths in each observation (experiment with inclusion of other moments and/or max pooling)
		# print ('Averaging over paths')
		# print (f'enc1.shape: {enc1.shape}')
		enc = torch.mean(enc1, 1)
		# print (f'enc.shape: {enc.shape}\n\n')


		# Flatten each observation's averaged time series
		# print ('Flattening avg time series')
		# print (f'enc.shape: {enc.shape}')
		# enc = torch.flatten(enc, start_dim=1)
		# print (f'enc.shape: {enc.shape}\n\n')

		# Prediction head
		# print ('MLP prediction head')
		# print (f'enc.shape: {enc.shape}')
		out = self.mlp_head(enc)
		# print (f'out.shape: {out.shape}')
		# print (out, '\n\n')


		if self.task == 'kcat regression':
			return out
		elif self.task == 'NR/R binary classification':
			return self.sigmoid(out)
		elif self.task == 'S/F binary classification':
			return self.sigmoid(out)



"""
PyTorch class for adding positional encodings to input data; typically
used inside transformer architectures (see class TransformerModel for an example)
"""
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
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
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



"""
Classes and functions related to calculating loss
"""
class DenseLoss(nn.Module):
    # Implements the DenseLoss loss function using DenseWeight
	# method (and object) as described here:
	# https://link.springer.com/article/10.1007/s10994-021-06023-5
	
	def __init__(self, dw):
		# INPUT:
		# dw -- an instance of DenseWeight that has been fit to data
		#    	by running its .fit() method
		super().__init__()
		self.dw = dw
	
	def forward(self, preds, targets):
		# Calculate relevance (weight) for each sample

		weights = torch.from_numpy( self.dw(targets.cpu().numpy()) ).to(device)

		# Calculate weighted MSE
		err = torch.pow(preds - targets, 2)
		err_weighted = weights * err
		mse = err_weighted.mean()

		return mse

def define_loss(model, dw=None):
	# Defines the loss function based on the model's prediction task, 
	# and whether loss weighting with DenseWeight is activated
	# INPUT:
	# model -- instance of TransformerModel
	# dw -- instance of DenseWeight or None. If None (default),
	#		then DenseWeight and DenseLoss are not applied, and 
	# 		observations' loss terms are weighted equally

	if (model.task == 'kcat regression') and (dw == None):
		loss_fn = nn.MSELoss()
	elif model.task == 'kcat regression':
		loss_fn = DenseLoss(dw)
	elif model.task == 'NR/R binary classification':
		loss_fn = nn.BCELoss()
	elif model.task == 'S/F binary classification':
		loss_fn = nn.BCELoss()

	return loss_fn

def plot_dw_alpha(pathdataset, alphas=[0, 0.5, 0.90, 0.95, 1.0], figsize=(16,8), figname=False):
	# Plot weights vs. log10(kcat) for different alpha
	# INPUT
	# pathdataset -- PathDataset object with the .obs attribute
	#				 populated. Or, a PyTorch DataLoader object 
	#				 constructed on a PathDataset object
	# alphas -- the DenseWeight alpha values with which to calculate
	#			weights for each kcat in pathdataset.obs.kcat
	# figsize -- figure size
	# figname -- file to which to save the plot

	if isinstance(pathdataset, PathDataset):
		kcats = np.log10(pathdataset.obs.kcat)

	elif isinstance(pathdataset, DataLoader):
		kcats = []
		for batch_idx, batch in enumerate(pathdataset):
			kcats.append(batch['kcat'].detach().cpu().numpy())
		kcats = np.concatenate(kcats)

	weights = {}
	df = pd.DataFrame(kcats, columns=['kcat'])
	df['bins'] = pd.cut(df['kcat'], bins=5)
	x = np.linspace(np.min(kcats), np.max(kcats), 100)
	for a in alphas:
		dw = DenseWeight(alpha=a, eps=1e-6, bandwidth=1)
		dw.fit(kcats)
		weights[a] = dw(x)
		df[f'weights (a={a})'] = dw(df['kcat'].to_numpy())

	# Plot
	fig, axes = plt.subplots(2, 3, figsize=figsize)
	for i,a in enumerate(weights):
		axes.flatten()[0].plot(x, weights[a], label=fr'$\alpha={a}$')
		
		ax = axes.flatten()[i+1]
		print (df)
		weight_sum = df.groupby(by='bins').apply(lambda x: np.sum(x[f'weights (a={a})'].to_numpy()))
		print (weight_sum)
		weight_sum.plot.bar(x='bins', ax=ax, title=fr'$\alpha={a}$')
		if (i+1) == len(weights):
			for tick in ax.get_xticklabels():
				tick.set_rotation(45)
				tick.set_fontsize=9
		else:
			ax.set_xticklabels(ax.get_xticks(), rotation=45)
		ax.set_ylabel('Sum of weights')
		print ('\n\n')

	axes.flatten()[0].hist(kcats, label=r'$p(log_{10}(k_{cat}))$', density=True)
	axes.flatten()[0].legend(fontsize=9)
	# axes.flatten()[0].set_xlabel(r'$log_{10}(k_{cat})$')
	fig.text(0.5, 0.02, r'$log_{10}(k_{cat})$', ha='center', fontsize=20)
	fig.tight_layout()
	fig.savefig(figname)

	return



"""
Functions related to learning rate schedules
"""
def warmup_decay_lr(step, d_model=256, warmup_steps=4000):
	# Learning rate schedule based on that used in Attention is All You Need:
	# https://arxiv.org/pdf/1706.03762.pdf
	# An adjustment was made to account for the lower-dimensional models
	# used in the default TransformerModel module. Specifically, the original 
	# schedule from Attention is All You Need is multiplied by a factor
	# of 1/sqrt(2) because the dimension of our default TransformerModel 
	# (dmodel) is half that of models used in Attention is All You Need.
	# INPUT:
	# step -- the training step count. Each training batch and update of 
	# 		  learned parameters is considered a single step. int
	# d_model -- the dimensions of the transformer encoder layers in the 
	#            transformer encoder. All sublayers in the model will produce
	#            outputs with this dimension. int
	# warmup_steps -- the number of warmup steps to use. int

	if step == 0:
		return 0

	term1 = step**-0.5
	term2 = step*warmup_steps**-1.5
	term = np.array([term1,term2])

	lr = (2*d_model)**-0.5 * np.min(term, axis=0)

	return lr



"""
Classes and functions related to model training and evaluation
"""
def train(model, dataloader, output_text) -> None:
	# Training function. Call this once for every epoch to run
	# through the data in dataloader and update the model parameters
	# INPUT:
	# model -- an instance of a PyTorch nn.Module
	# dataloader -- PyTorch DataLoader to stream training data
	# output_text -- open file to which output will be written

	model.train()
	total_loss, total_acc = 0.0, 0.0
	start_time = time.time()
	log_interval = 25 # print info every log_interval number of batches

	for batch_idx, batch in enumerate(dataloader):

		# forward pass
		output = model(batch['paths'])

		# print (type(output), output, output.dtype)
		# print (type(batch['kcat'].view(-1,1)), batch['kcat'].view(-1,1), batch['kcat'].view(-1,1).dtype)

		if model.task == 'kcat regression':
			loss = loss_fn(output, batch['kcat'].view(-1,1))
		elif model.task == 'NR/R binary classification':
			loss = loss_fn(output, batch['order'].view(-1,1))
			acc = (output.round() == batch['order'].view(-1,1)).float().mean()
			total_acc += acc
		elif model.task == 'S/F binary classification':
			loss = loss_fn(output, batch['kcat_class'].view(-1,1))
			acc = (output.round() == batch['kcat_class'].view(-1,1)).float().mean()
			total_acc += acc

		# backward pass
		optimizer.zero_grad()
		loss.backward()
		# torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) # uncomment to help prevent gradients from exploding
		# more info: https://pytorch.org/tutorials/beginner/transformer_tutorial.html

		# update weights
		optimizer.step()

		# update loss 
		total_loss += loss.item()


		if ((batch_idx+1) % log_interval == 0) or ((batch_idx+1) == len(dataloader)):
			
			# print progress and info
			time_per_batch = (time.time() - start_time) / log_interval
			current_loss = total_loss / log_interval
			current_acc = total_acc / log_interval # only reported when task is '* classification'
			root_loss = np.sqrt(current_loss)
			curr_lr = scheduler.get_last_lr()[0]

			if model.task == 'kcat regression':
				print (f'epoch {epoch} | CV fold {cv_fold}/{cv_folds} | {batch_idx+1:d}/{len(dataloader):d} batches | '
						f'lr {curr_lr:0.3e} | seconds/batch {time_per_batch:.3f} | '
						f'loss {current_loss:.3e} | sqrt(loss) {root_loss:.3e}')
				output_text.write(f'epoch {epoch} | CV fold {cv_fold}/{cv_folds} | {batch_idx+1:d}/{len(dataloader):d} batches | '
						f'lr {curr_lr:0.3e} | seconds/batch {time_per_batch:.3f} | '
						f'loss {current_loss:.3e} | sqrt(loss) {root_loss:.3e}\n')

			elif model.task in ['NR/R binary classification', 'S/F binary classification']:
				print (f'epoch {epoch} | CV fold {cv_fold}/{cv_folds} | {batch_idx+1:d}/{len(dataloader):d} batches | '
						f'lr {curr_lr:0.3e} | seconds/batch {time_per_batch:.3f} | '
						f'loss {current_loss:.3e} | sqrt(loss) {root_loss:.3e} | '
						f'acc {current_acc:.5f}')
				output_text.write(f'epoch {epoch} | CV fold {cv_fold}/{cv_folds} | {batch_idx+1:d}/{len(dataloader):d} batches | '
						f'lr {curr_lr:0.3e} | seconds/batch {time_per_batch:.3f} | '
						f'loss {current_loss:.3e} | sqrt(loss) {root_loss:.3e} | '
						f'acc {current_acc:.5f}\n')

				total_acc = 0.0

			output_text.flush()

			# reset loss and timer for next round of log_interval number of batches
			total_loss = 0.0
			start_time = time.time()


		# update learning rate
		scheduler.step()

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
	total_loss, total_acc = 0.0, 0.0
	total_obs = 0
	with torch.no_grad():
		for batch_idx, batch in enumerate(dataloader):
			n_obs = batch['paths'].size(0)

			output = model(batch['paths'])

			if model.task == 'kcat regression':
				loss = loss_fn(output, batch['kcat'].view(-1,1))

			elif model.task == 'NR/R binary classification':
				loss = loss_fn(output, batch['order'].view(-1,1))
				acc = (output.round() == batch['order'].view(-1,1)).float().mean()
				total_acc += acc * n_obs

			elif model.task == 'S/F binary classification':
				loss = loss_fn(output, batch['kcat_class'].view(-1,1))
				acc = (output.round() == batch['kcat_class'].view(-1,1)).float().mean()
				total_acc += acc * n_obs

			total_loss += loss.item() * n_obs
			total_obs += n_obs



	avg_loss = total_loss / total_obs # this is equivalent to MSE when doing 'kcat regression' task
	avg_acc  = total_acc / total_obs # this will be 0 when doing 'kcat regression' task

	return {'total loss':total_loss, 'avg loss':avg_loss, 'avg acc':avg_acc}

class Fold():
	# Object for storing information specific to different CV folds and 
    # for executing training and testing loops for a given fold

	def __init__(self, cv_fold, train_idx, test_idx, parallel_folds, 
	      		 output_text_filename, model_kwargs, data, device, 
				 stoch_labels, dw_settings, control_model, epochs,
				 random_seed, batch_size, cv_folds):
		self.cv_fold = cv_fold
		self.train_idx = train_idx
		self.test_idx = test_idx
		self.parallel_folds = parallel_folds
		# self.output_text = output_text ### This is probably the issue preventing pickling
		self.output_text_filename = output_text_filename
		self.model_kwargs = model_kwargs
		self.data = data
		self.device = device
		self.stoch_labels = stoch_labels
		self.dw_settings = dw_settings
		self.control_model = control_model
		self.epochs = epochs
		self.random_seed = random_seed
		self.batch_size = batch_size
		self.cv_folds = cv_folds
		
	def run(self):
        # If running folds in parallel, give each a unique output file
		if not self.parallel_folds:
			self.fold_output_text = open(self.output_text_filename, 'a')
		else:
			self.fold_output_text = open(f'{self.output_text_filename}-fold{self.cv_fold}', 'w')
		
		# Define a new model
		self.model = TransformerModel(**self.model_kwargs).to(self.device)
		
		total_params = sum(p.numel() for p in self.model.parameters())
		trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
		print (f'Total parameters:     {total_params}')
		print (f'Trainable parameters: {trainable_params}')
		self.fold_output_text.write(f'Total parameters:     {total_params}\n')
		self.fold_output_text.write(f'Trainable parameters: {trainable_params}\n')
		
		# Fit the scaler to the training data
		scaler = NormalScaler()
		scaler.fit(self.data.data[np.concatenate(self.data.obs.obs[self.train_idx]),:,:])
		train_data_scaler = DataScaler(scaler, stoch_labels=self.stoch_labels)
		test_data_scaler  = DataScaler(scaler, stoch_labels=False)
		# print ('\n\n\n\n Finished fitting SCALER \n\n\n\n')

		
		# Fit weighting function to the training data, if weighted loss is activated
		if self.dw_settings != None:
			if task != 'kcat regression':
				raise ValueError("DenseLoss is only implemented for task='kcat regression'."+\
                                 "You cannot specify dw_settings for other tasks.")
			dw = DenseWeight(alpha=self.dw_settings['alpha'],
							 bandwidth=self.dw_settings['bandwidth'],
							 eps=self.dw_settings['eps'])
			dw.fit(np.log10(self.data.obs.kcat[self.train_idx]))
		else: 
			dw = None
				
		# Define loss, optimizer, and a learning rate scheduler
		self.loss_fn = define_loss(self.model, dw)
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1) # lr will be scaled and set by scheduler
		self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, warmup_decay_lr)
		
		# Define datasets and dataloaders for train and test sets
		train_dataset = PathTorchDataset(self.data, elligible_idxs=self.train_idx, transform=train_data_scaler, control_model=self.control_model)
		test_dataset  = PathTorchDataset(self.data, elligible_idxs=self.test_idx, transform=test_data_scaler)
		train_variants = np.unique(train_dataset.pathdataset.obs.variant[train_dataset.elligible_idxs])
		test_variants = np.unique(test_dataset.pathdataset.obs.variant[test_dataset.elligible_idxs])
		
		print ('\n\nTraining variants:')
		print (train_variants, '\n\n')
		self.fold_output_text.write('\n\nTraining variants:\n')
		self.fold_output_text.write(str(train_variants))
		self.fold_output_text.write('\n\n\n')
		self.fold_output_text.flush()
		
		best_val_loss = float('inf')
		best_model_file = f'best_model_cvfold{self.cv_fold}.pt'
		for epoch in range(1, (self.epochs+1)):
			
			self.epoch = epoch
			start_epoch = time.time()
			
			# initialize data loading, explicitly seed random number generators for reproducibility
			train_rng, test_rng = torch.Generator(), torch.Generator()
			train_rng.manual_seed(int(self.random_seed*1e10/41/epoch*self.cv_fold))
			test_rng.manual_seed(int(self.random_seed*1e10/79/epoch*self.cv_fold))
			self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, generator=train_rng)
			self.test_loader  = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True, generator=test_rng)
			
			# train and test model
			self.train()
			val_results = self.evaluate()
			
			# report results
			epoch_duration = time.time() - start_epoch
			loss, mse = val_results['total loss'], val_results['avg loss']
			acc, auroc = val_results['avg acc'], val_results['auroc']
			print ('\n\n','-'*80)
			self.fold_output_text.write('\n\n'+'-'*80+'\n')
			
			if self.model.task == 'kcat regression':
				print (f'End of epoch {self.epoch} | CV fold {self.cv_fold}/{self.cv_folds} | '
                       f'time: {epoch_duration:.1f}s | '
                       f'valid loss (MSE) {mse:.3f} | RMSE {mse**0.5:.3f}')
				
				self.fold_output_text.write(f'End of epoch {self.epoch} | CV fold {self.cv_fold}/{self.cv_folds} | '
                                       f'time {epoch_duration:.1f}s | '
                                       f'valid loss (MSE) {mse:.3f} | RMSE {mse**0.5:.3f}\n')
			
			elif self.model.task in ['NR/R binary classification', 'S/F binary classification']:
				print (f'End of epoch {self.epoch} | CV fold {self.cv_fold}/{self.cv_folds} | '
                       f' time: {epoch_duration:.1f}s | '
                       f'valid loss {mse:.3f} | '
                       f'accuracy {acc:.5f} | '
					   f'auroc {auroc:.5f}')
				
				self.fold_output_text.write(f'End of epoch {self.epoch} | CV fold {self.cv_fold}/{self.cv_folds} | '
                                    f'time {epoch_duration:.1f}s | '
                                    f'valid loss {mse:.3f} | '
				    				f'accuracy {acc:.5f} | '
                                    f'auroc {auroc:.5f}\n')
			
			print ('\nValidation variants:')
			print (test_variants)
			self.fold_output_text.write('\nValidation variants:\n')
			self.fold_output_text.write(str(test_variants))
			self.fold_output_text.write('\n')
			
			if mse < best_val_loss:
				print (f'\nBest loss achieved, saving model state to {best_model_file}')
				self.fold_output_text.write(f'\nBest loss achieved, saving model state to {best_model_file}\n')
				best_val_loss = mse
				torch.save(self.model.state_dict(), best_model_file)
                # to load into model again later:
                # model.load_state_dict(torch.load(best_model_file))
			
			print ('-'*80, '\n\n')
			self.fold_output_text.write('-'*80+'\n\n\n')
			self.fold_output_text.flush()
		
		# if parallel_folds:
		self.fold_output_text.close()

	def train(self) -> None:
		# Training function. Call this once for every epoch to run
		# through the data in dataloader and update the model parameters
		# INPUT:
		# model -- an instance of a PyTorch nn.Module
		# dataloader -- PyTorch DataLoader to stream training data
		# output_text -- open file to which output will be written

		self.model.train()
		total_loss, total_acc = 0.0, 0.0
		start_time = time.time()
		log_interval = 25 # print info every log_interval number of batches
		batch_count = 0

		for batch_idx, batch in enumerate(self.train_loader):

			# forward pass
			output = self.model(batch['paths'])

			# print (type(output), output, output.dtype)
			# print (type(batch['kcat'].view(-1,1)), batch['kcat'].view(-1,1), batch['kcat'].view(-1,1).dtype)

			if self.model.task == 'kcat regression':
				loss = self.loss_fn(output, batch['kcat'].view(-1,1))
			elif self.model.task == 'NR/R binary classification':
				loss = self.loss_fn(output, batch['order'].view(-1,1))
				acc = (output.round() == batch['order'].view(-1,1)).float().mean()
				total_acc += acc
			elif self.model.task == 'S/F binary classification':
				loss = self.loss_fn(output, batch['kcat_class'].view(-1,1))
				acc = (output.round() == batch['kcat_class'].view(-1,1)).float().mean()
				total_acc += acc

			# backward pass
			self.optimizer.zero_grad()
			loss.backward()
			# torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) # uncomment to help prevent gradients from exploding
			# more info: https://pytorch.org/tutorials/beginner/transformer_tutorial.html

			# update weights
			self.optimizer.step()

			# update loss 
			total_loss += loss.item()

			# update batch count, which is reset to 0 after printing progress
			batch_count += 1


			if ((batch_idx+1) % log_interval == 0) or ((batch_idx+1) == len(self.train_loader)):
				
				# print progress and info
				time_per_batch = (time.time() - start_time) / batch_count
				current_loss = total_loss / batch_count
				current_acc = total_acc / batch_count # only reported when task is '* classification'
				root_loss = np.sqrt(current_loss)
				curr_lr = self.scheduler.get_last_lr()[0]

				if self.model.task == 'kcat regression':
					print (f'epoch {self.epoch} | CV fold {self.cv_fold}/{self.cv_folds} | {batch_idx+1:d}/{len(self.train_loader):d} batches | '
							f'lr {curr_lr:0.3e} | seconds/batch {time_per_batch:.3f} | '
							f'loss {current_loss:.3e} | sqrt(loss) {root_loss:.3e}')
					self.fold_output_text.write(f'epoch {self.epoch} | CV fold {self.cv_fold}/{self.cv_folds} | {batch_idx+1:d}/{len(self.train_loader):d} batches | '
							f'lr {curr_lr:0.3e} | seconds/batch {time_per_batch:.3f} | '
							f'loss {current_loss:.3e} | sqrt(loss) {root_loss:.3e}\n')

				elif self.model.task in ['NR/R binary classification', 'S/F binary classification']:
					print (f'epoch {self.epoch} | CV fold {self.cv_fold}/{self.cv_folds} | {batch_idx+1:d}/{len(self.train_loader):d} batches | '
							f'lr {curr_lr:0.3e} | seconds/batch {time_per_batch:.3f} | '
							f'loss {current_loss:.3e} | sqrt(loss) {root_loss:.3e} | '
							f'acc {current_acc:.5f}')
					self.fold_output_text.write(f'epoch {self.epoch} | CV fold {self.cv_fold}/{self.cv_folds} | {batch_idx+1:d}/{len(self.train_loader):d} batches | '
							f'lr {curr_lr:0.3e} | seconds/batch {time_per_batch:.3f} | '
							f'loss {current_loss:.3e} | sqrt(loss) {root_loss:.3e} | '
							f'acc {current_acc:.5f}\n')

					total_acc = 0.0

				self.fold_output_text.flush()

				# reset loss, timer, and batch_count for next round of log_interval number of batches
				total_loss = 0.0
				start_time = time.time()
				batch_count = 0


			# update learning rate
			self.scheduler.step()


	def evaluate(self) -> float:
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

		self.model.eval()
		total_loss, total_acc = 0.0, 0.0
		total_obs = 0
		total_preds, total_labels = [], []
		with torch.no_grad():
			for batch_idx, batch in enumerate(self.test_loader):
				n_obs = batch['paths'].size(0)

				output = self.model(batch['paths'])

				if self.model.task == 'kcat regression':
					labels = batch['kcat'].view(-1,1)
					loss = self.loss_fn(output, labels)

				elif self.model.task == 'NR/R binary classification':
					labels = batch['order'].view(-1,1)
					loss = self.loss_fn(output, labels)
					acc = (output.round() == labels).float().mean()
					total_acc += acc * n_obs

				elif self.model.task == 'S/F binary classification':
					labels = batch['kcat_class'].view(-1,1)
					loss = self.loss_fn(output, labels)
					acc = (output.round() == labels).float().mean()
					total_acc += acc * n_obs

				total_loss += loss.item() * n_obs
				total_obs += n_obs

				total_preds.append(output.cpu().numpy())
				total_labels.append(labels.cpu().numpy())

		# If doing classification, report the AUROC
		if 'classification' in self.model.task:
			auroc = AUROC(np.concatenate(total_labels), np.concatenate(total_preds))
		else:
			auroc = None

		avg_loss = total_loss / total_obs # this is equivalent to MSE when doing 'kcat regression' task
		avg_acc  = total_acc / total_obs # this will be 0 when doing 'kcat regression' task

		return {'total loss':total_loss, 'avg loss':avg_loss, 'avg acc':avg_acc, 'auroc':auroc}

def run_fold(cv_fold, train_idx, test_idx):

    # If running folds in parallel, give each a unique output file
    if not parallel_folds:
        fold_output_text = output_text
    else:
        fold_output_text = open(f'{output_text_filename}-fold{cv_fold}', 'w')

    # Define a new model
    model = TransformerModel(input_size=data.data.shape[-1],
                                input_length=data.data.shape[-2],
                                d_model=d_model,
                                n_head=n_head,
                                d_tran_ffn=d_tran_ffn,
                                dropout_tran_encoder=dropout_tran_encoder,
                                n_tran_layers=n_tran_layers,
                                d_mlp_head=d_mlp_head,
                                dropout_mlp_head=dropout_mlp_head,
                                task=task).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print (f'Total parameters:     {total_params}')
    print (f'Trainable parameters: {trainable_params}')


    # Fit the scaler to the training data
    scaler = NormalScaler()
    scaler.fit(data.data[np.concatenate(data.obs.obs[train_idx]),:,:])
    train_data_scaler = DataScaler(scaler, stoch_labels=stoch_labels)
    test_data_scaler  = DataScaler(scaler, stoch_labels=False)
    # print ('\n\n\n\n Finished fitting SCALER \n\n\n\n')


    # Fit weighting function to the training data, if weighted loss is activated
    if dw_settings != None:
        if task != 'kcat regression':
            raise ValueError("DenseLoss is only implemented for task='kcat regression'."+\
                                "You cannot specify dw_settings for other tasks.")
        dw = DenseWeight(alpha=dw_settings['alpha'],
                            bandwidth=dw_settings['bandwidth'],
                            eps=dw_settings['eps'])
        dw.fit(np.log10(data.obs.kcat[train_idx]))
    else: 
        dw = None


    # Define loss, optimizer, and a learning rate scheduler
    loss_fn = define_loss(model, dw)
    optimizer = torch.optim.Adam(model.parameters(), lr=1) # lr will be scaled and set by scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_decay_lr)


    # Define datasets and dataloaders for train and test sets
    train_dataset = PathTorchDataset(data, elligible_idxs=train_idx, transform=train_data_scaler, control_model=control_model)
    test_dataset  = PathTorchDataset(data, elligible_idxs=test_idx, transform=test_data_scaler)
    train_variants = np.unique(train_dataset.pathdataset.obs.variant[train_dataset.elligible_idxs])
    test_variants = np.unique(test_dataset.pathdataset.obs.variant[test_dataset.elligible_idxs])

    print ('\n\nTraining variants:')
    print (train_variants, '\n\n')
    fold_output_text.write('\n\nTraining variants:\n')
    fold_output_text.write(str(train_variants))
    fold_output_text.write('\n\n\n')
    fold_output_text.flush()

    best_val_loss = float('inf')
    best_model_file = f'best_model_cvfold{cv_fold}.pt'
    for epoch in range(1, (epochs+1)):

        start_epoch = time.time()

        # initialize data loading, explicitly seed random number generators for reproducibility
        train_rng, test_rng = torch.Generator(), torch.Generator()
        train_rng.manual_seed(int(random_seed*1e10/41/epoch))
        test_rng.manual_seed(int(random_seed*1e10/79/epoch))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=train_rng)
        test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, generator=test_rng)

        # train and test model
        train(model, train_loader, fold_output_text)
        val_results = evaluate(model, test_loader)

        # report results
        epoch_duration = time.time() - start_epoch
        loss, mse, acc = val_results['total loss'], val_results['avg loss'], val_results['avg acc']
        print ('\n\n','-'*80)
        fold_output_text.write('\n\n'+'-'*80+'\n')

        if model.task == 'kcat regression':
            print (f'End of epoch {epoch} | CV fold {cv_fold}/{cv_folds} | '
                    f'time: {epoch_duration:.1f}s | '
                    f'valid loss (MSE) {mse:.3f} | RMSE {mse**0.5:.3f}')

            fold_output_text.write(f'End of epoch {epoch} | CV fold {cv_fold}/{cv_folds} | '
                                f'time {epoch_duration:.1f}s | '
                                f'valid loss (MSE) {mse:.3f} | RMSE {mse**0.5:.3f}\n')

        elif model.task in ['NR/R binary classification', 'S/F binary classification']:
            print (f'End of epoch {epoch} | CV fold {cv_fold}/{cv_folds} | '
                    f' time: {epoch_duration:.1f}s | '
                    f'valid loss {mse:.3f} | '
                    f'accuracy {acc:.5f}')

            fold_output_text.write(f'End of epoch {epoch} | CV fold {cv_fold}/{cv_folds} | '
                                f'time {epoch_duration:.1f}s | '
                                f'valid loss {mse:.3f} | '
                                f'accuracy {acc:.5f}\n')
        
        print ('\nValidation variants:')
        print (test_variants)
        fold_output_text.write('\nValidation variants:\n')
        fold_output_text.write(str(test_variants))
        fold_output_text.write('\n')

        if mse < best_val_loss:
            print (f'\nBest loss achieved, saving model state to {best_model_file}')
            fold_output_text.write(f'\nBest loss achieved, saving model state to {best_model_file}\n')
            best_val_loss = mse
            torch.save(model.state_dict(), best_model_file)
            # to load into model again later:
            # model.load_state_dict(torch.load(best_model_file))

        print ('-'*80, '\n\n')
        fold_output_text.write('-'*80+'\n\n\n')
        fold_output_text.flush()

    if parallel_folds:
        fold_output_text.close()



"""
Classes for further, more in-depth evaluation of models post-training
"""
class LogFile():
    """
    Given the output_text file filename (str) generated by transformer_1.py, 
    this class parses the file and contains functions to plot model performance 
    during training. Typical usage to plot RMSE across batches and epochs:
    log = LogFile('transformer_1_output.txt')
    log.plot_summary()
    """
    
    def __init__(self, filename, figname=False):
        # INPUT
        # filename -- name of output_text file. str
        # figname -- name of file to save output figure to
        self.task = 'kcat regression' # default task if unspecified in config file
        self.filename = filename
        self.config = glob(f'{os.path.dirname(os.path.abspath(self.filename))}/*config*')[0]
        with open(self.config, 'r') as f:
            settings = f.read()
        if 'NR/R binary classification' in settings:
            self.task = 'NR/R binary classification'
        elif 'kcat regression' in settings:
            self.task = 'kcat regression'
        elif 'S/F binary classification' in settings:
            self.task = 'S/F binary classification'

        if self.task == 'kcat regression':
            self.epoch_cols = ['Epoch', 'CV fold', 'Time', 'Val. loss', 'RMSE']
            self.batch_cols = ['Epoch', 'CV fold', 'Batch', 'Learning rate', 'Time/batch', 'Loss (MSE)', 'RMSE']
        elif self.task in ['NR/R binary classification', 'S/F binary classification']:
            self.epoch_cols = ['Epoch', 'CV fold', 'Time', 'Val. loss', 'Acc.', 'AUROC']
            self.batch_cols = ['Epoch', 'CV fold', 'Batch', 'Learning rate', 'Time/batch', 'Loss', 'sqrt(loss)', 'Acc.', 'AUROC']

        self.figname = figname
        
        batches, epochs = [], []
        with open(self.filename, 'r') as f:
            
            for line in f:
                if len(line.rsplit()) == 0: continue
                if line.rsplit()[0] == 'epoch':
                    batches.append(line)
                elif line.rsplit()[0] == 'End':
                    epochs.append(line)
        
        # store epochs log in DataFrame
        tmp = tempfile.TemporaryFile(mode='w+t')
        tmp.writelines(epochs)
        tmp.seek(0)
        self.epochs = self.clean_epoch_table(pd.read_csv(tmp, sep='|', names=self.epoch_cols))
        tmp.close()
        
        # store batches log in DataFrame
        tmp = tempfile.TemporaryFile(mode='w+t')
        tmp.writelines(batches)
        tmp.seek(0)
        self.batches = self.clean_batch_table(pd.read_csv(tmp, sep='|', names=self.batch_cols))
        tmp.close()
    
    def clean_epoch_table(self, epochs):
        
        epochs['Epoch'] = [x.replace('End of epoch ','') for x in epochs['Epoch']]
        epochs['CV fold'] = [x.split('fold ')[-1].split('/')[0] for x in epochs['CV fold']]
        epochs['Time'] = [x.split('time ')[-1].split('s')[0] for x in epochs['Time']]

        if self.task == 'kcat regression':
            epochs['Val. loss'] = [x.split('MSE) ')[-1] for x in epochs['Val. loss']]
            epochs['RMSE'] = [x.split('RMSE ')[-1].split('s')[0] for x in epochs['RMSE']]
        elif self.task in ['NR/R binary classification', 'S/F binary classification']:
            epochs['Val. loss'] = [x.split('valid loss ')[-1] for x in epochs['Val. loss']]
            epochs['Acc.'] = [x.split('accuracy ')[-1] for x in epochs['Acc.']]
            if str(epochs['AUROC'][0]) == 'nan':
                # AUROC wasn't calculated for this job
                epochs.drop(columns=['AUROC'])
            else:
                epochs['AUROC'] = [x.split('auroc ')[-1] for x in epochs['AUROC']]
        
        return epochs.apply(pd.to_numeric)
    
    def clean_batch_table(self, batches):
        
        batches['Epoch'] = [x.replace('epoch ','') for x in batches['Epoch']]
        batches['CV fold'] = [x.split('fold ')[-1].split('/')[0] for x in batches['CV fold']]
        batches['Batch'] = [x.split('/')[0] for x in batches['Batch']]
        batches['Learning rate'] = [x.split('lr ')[-1] for x in batches['Learning rate']]
        batches['Time/batch'] = [x.split('batch ')[-1] for x in batches['Time/batch']]

        if self.task == 'kcat regression':
            batches['Loss (MSE)'] = [x.split('loss ')[-1] for x in batches['Loss (MSE)']]
            batches['RMSE'] = [x.split('loss) ')[-1] for x in batches['RMSE']]
        elif self.task in ['NR/R binary classification', 'S/F binary classification']:
            batches['Loss'] = [x.split('loss ')[-1] for x in batches['Loss']]
            batches['sqrt(loss)'] = [x.split('loss) ')[-1] for x in batches['sqrt(loss)']]
            batches['Acc.'] = [x.split('acc ')[-1] for x in batches['Acc.']]
            if str(batches['AUROC'][0]) == 'nan':
                # AUROC wasn't calculated for this job
                batches.drop(columns=['AUROC'])
            else:
                batches['AUROC'] = [x.split('auroc ')[-1] for x in batches['AUROC']]

        return batches.apply(pd.to_numeric)
        
    def plot_batches(self, ax=None):
        if self.task == 'kcat regression':
            ax = sns.lineplot(data=self.batches, x='Batch', y='RMSE', hue='Epoch', palette='flare', ax=ax)
            ax.set_ylabel('RMSE (train)', fontsize=20)
        elif self.task in ['NR/R binary classification', 'S/F binary classification']:
            ax = sns.lineplot(data=self.batches, x='Batch', y='Acc.', hue='Epoch', palette='flare', ax=ax)
            ax.set_ylabel('Accuracy (train)', fontsize=20) 

        ax.set_xlabel('Batch', fontsize=20)
        ax.tick_params(axis='both', labelsize=18)
        return ax
        
    def plot_epochs(self, ax=None):
        if self.task == 'kcat regression':
            ax = sns.lineplot(data=self.epochs, x='Epoch', y='RMSE', hue='CV fold', palette='flare', ax=ax)
            ax.set_ylabel('RMSE (validation)', fontsize=20)
        elif self.task in ['NR/R binary classification', 'S/F binary classification']:
            ax = sns.lineplot(data=self.epochs, x='Epoch', y='Acc.', hue='CV fold', palette='flare', ax=ax)
            ax.set_ylabel('Accuracy (validation)', fontsize=20)

        ax.set_xlabel('Epoch', fontsize=20)
        ax.tick_params(axis='both', labelsize=18)
        return ax

    def plot_summary(self):
        fig, axes = plt.subplots(3,1,figsize=(6,15))
        axes.flatten()[0] = self.plot_batches(ax=axes.flatten()[0])
        axes.flatten()[1] = self.plot_batches(ax=axes.flatten()[1])
        if self.task == 'kcat regression':
            axes.flatten()[1].set_ylim((0,2))
        elif self.task == 'NR/R binary classification':
            axes.flatten()[1].set_ylim((0.98,1))
        axes.flatten()[2] = self.plot_epochs(ax=axes.flatten()[2])
        fig.tight_layout()
        if self.figname:
            fig.savefig(self.figname, dpi=300)
        plt.show()
        return fig, axes
	
    def cv_summary(self, log=False, shade_err=False):
        # log sets log scale for y-axis when True
        # shade_err shades +/- 1STD across batches in each epoch for training loss
        if self.task == 'kcat regression':
            metric = 'Loss (MSE)'
        else:
            metric = 'Loss'

        n_folds = np.unique(self.epochs['CV fold']).shape[0]
        if n_folds in [5,10]:
            ncols = 5
        else:
            ncols = 3
        nrows = int(np.ceil(n_folds/ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(6.4*ncols,4.8*nrows))
		
        for i,fold in enumerate(np.arange(n_folds)+1):
            tmp_batches = self.batches.loc[self.batches['CV fold'] == fold]
            avg_batches = tmp_batches.groupby('Epoch').mean()
            std_batches = tmp_batches.groupby('Epoch').std()
			
            tmp_epochs = self.epochs.loc[self.epochs['CV fold'] == fold]
            ax = axes.flatten()[i]
            ax.plot(avg_batches.index, avg_batches[metric], label='Train', color='k', alpha=0.8)
            if shade_err:
                ax.fill_between(std_batches.index, avg_batches[metric]-std_batches[metric], avg_batches[metric]+std_batches[metric], color='k', alpha=0.3)
            ax.plot(tmp_epochs['Epoch'], tmp_epochs['Val. loss'], label='Test', color='b', alpha=0.8)

            ax.set_xlabel('Epoch', fontsize=20)
            ax.set_ylabel('Loss', fontsize=20)
            ax.set_title(f'Fold {fold}', fontsize=20)
            ax.tick_params(axis='both', labelsize=16)
            ax.legend(fontsize=20)

            if log:
                ax.set_yscale('log')

        fig.tight_layout()
        if self.figname:
            fig.savefig(self.figname, dpi=300)
        plt.show()

class ModelTest(PathDataset):
    # Class that organizes a model and dataset for execution of
    # various model performance analyses
    
    def __init__(self, model_file, data_file, meta_file=None, output_file=None, scaler=None):
        
        super().__init__(data_file, meta_file, path_set_size=1)
        self.scaler = scaler
        self.model_file = model_file
        self.config_file = glob('/'.join(model_file.split('/')[0:-1]) + '/*conf*txt')[0]
        self.train_vars, self.test_vars = ModelTest.get_val_variants(model_file, output_file=output_file)
        
        # Create train indexes corresponding to variants in the train set
        train_idx = np.nonzero(np.in1d(np.array(self.meta.variant),np.unique(self.train_vars)))[0]

        # Load the model
        #self.model = TransformerModel(self.data.shape[-1], self.data.shape[-2], d_model=128) ### read d_model from config file
        self.model = self.make_model()
        self.model.load_state_dict(torch.load(self.model_file))

        # Fit the scaler to the training data
        if self.scaler == None:
            self.scaler = NormalScaler()
            self.scaler.fit(self.data[train_idx,:,:])
            self.data_scaler = DataScaler(self.scaler)
        else:
            self.data_scaler = DataScaler(self.scaler)
        
        # Initialize some variables that can later be populated with data
        self.variants, self.preds, self.targets = None, None, None
        
    def make_model(self):
        # Creates the PyTorch model object into which pre-trained
        # weights are loaded
        d_model = 256
        n_head = 4
        d_tran_ffn = 1024
        dropout_tran_encoder = 0.2
        n_tran_layers = 2
        d_mlp_head = 128
        dropout_mlp_head = 0.2
        with open(self.config_file, 'r') as f:
            settings = f.read()
            exec(settings)
        model = TransformerModel(input_size=self.data.shape[-1],
                                 input_length=self.data.shape[-2],
                                d_model = 128,
                                n_head = 2,
                                d_tran_ffn = 256,
                                dropout_tran_encoder = 0.2,
                                n_tran_layers = 1,
                                d_mlp_head = 64,
                                dropout_mlp_head = 0.2).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        
        return model
    
    def score_var(self, var):
        
        var_idx = np.nonzero(np.in1d(np.array(self.obs.variant), var))[0]
        var_dataset = PathTorchDataset(self, elligible_idxs=var_idx, transform=self.data_scaler)

        loader = DataLoader(var_dataset, batch_size=32, shuffle=True)

        self.model.eval()
        total_obs = 0
        with torch.no_grad():
            
            results = []
            for batch_idx, batch in enumerate(loader):

                n_obs = batch['paths'].size(0)
                output = self.model(batch['paths']).cpu().numpy().reshape((-1))
                total_obs += n_obs
                # self.results[var][path_set_size].append(output)
                results.append(output)
    
        return results
        
    def plot_test_var_pred(self, path_set_sizes=[10,100,1000], figname=False):
        # Plots the distribution of predicted log(kcat) values 
        # for each held-out (i.e., validation) variant as 
        # function of the path_set_size, which is the number of
        # paths included in each observation of the variant
        
        results = {}
        for var in self.test_vars:
            d = {}
            for size in path_set_sizes:
                d[size] = []
            results[var] = d
            
        for path_set_size in path_set_sizes:
            
            print (f'Testing path_set_size: {path_set_size}')
            self.path_set_size = path_set_size
            
            # Populate the obs attribute
            self.make_observations()
            
            # grab indexes of the held-out variants
            for i,var in enumerate(np.unique(self.test_vars)):
                res = self.score_var(var)
                results[var][path_set_size] = res
                print (f'Completed {i+1}/{self.test_vars.shape[0]} variants...')
            print ()
            
            # Empty the obs attribute to reset it
            self.obs = None     
            
        for var in results:
            for size in results[var]:
                results[var][size] = np.concatenate(results[var][size])
                
        # Plot
        ncols = 3
        nrows = int(np.ceil(len(results)/ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(9*ncols,6*nrows))
        for i,var in enumerate(results):

            var_kcat = np.log10(self.meta.kcat[np.where(np.array(self.meta.variant) == var)[0][0]])

            path_set_sizes, y_pred = [],[]
            for path_set_size in results[var]:

                data = results[var][path_set_size]
                y_pred += list(data)
                path_set_sizes += [path_set_size]*data.shape[0]


            df = pd.DataFrame({'path_set_size': path_set_sizes, 'Pred. log(kcat)': y_pred})

            # Create violin plot with seaborn
            axes.flatten()[i] = sns.violinplot(data=df, x='path_set_size', y='Pred. log(kcat)', ax=axes.flatten()[i])
            axes.flatten()[i].axhline(var_kcat, ls='--', c='gray', label=r'TIS log($k_{cat}$)')
            axes.flatten()[i].set_xlabel('')
            axes.flatten()[i].set_ylabel('')
            axes.flatten()[i].set_title(var, fontsize=22)
            axes.flatten()[i].legend(fontsize=20)
            axes.flatten()[i].tick_params(axis='both', labelsize=22)
            fig.text(0.5, 0.07, 'Paths per prediction', ha='center', fontsize=32)
            fig.text(0.05, 0.5, r'Predicted log($k_{cat}$)', va='center', rotation='vertical', fontsize=32)

        # fig.tight_layout()
        if figname:
            fig.savefig(figname, dpi=300)
        
        return results
    
    def plot_train_var_pred(self, path_set_sizes=[10,100,1000], figname=False):
        # Plots the distribution of predicted log(kcat) values 
        # for each variant included during training as a 
        # function of the path_set_size, which is the number of
        # paths included in each observation of the variant
        
        results = {}
        for var in self.train_vars:
            d = {}
            for size in path_set_sizes:
                d[size] = []
            results[var] = d
            
        for path_set_size in path_set_sizes:
            
            print (f'Testing path_set_size: {path_set_size}')
            self.path_set_size = path_set_size
            
            # Populate the obs attribute
            self.make_observations()
            
            # grab indexes of the held-out variants
            for i,var in enumerate(np.unique(self.train_vars)):
                res = self.score_var(var)
                results[var][path_set_size] = res
                print (f'Completed {i+1}/{self.train_vars.shape[0]} variants...')
            print ()
            
            # Empty the obs attribute to reset it
            self.obs = None     
            
        for var in results:
            for size in results[var]:
                results[var][size] = np.concatenate(results[var][size])
                
        # Plot
        ncols = 3
        nrows = int(np.ceil(len(results)/ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(9*ncols,6*nrows))
        for i,var in enumerate(results):

            var_kcat = np.log10(self.meta.kcat[np.where(np.array(self.meta.variant) == var)[0][0]])

            path_set_sizes, y_pred = [],[]
            for path_set_size in results[var]:

                data = results[var][path_set_size]
                y_pred += list(data)
                path_set_sizes += [path_set_size]*data.shape[0]


            df = pd.DataFrame({'path_set_size': path_set_sizes, 'Pred. log(kcat)': y_pred})

            # Create violin plot with seaborn
            axes.flatten()[i] = sns.violinplot(data=df, x='path_set_size', y='Pred. log(kcat)', ax=axes.flatten()[i])
            axes.flatten()[i].axhline(var_kcat, ls='--', c='gray', label=r'TIS log($k_{cat}$)')
            axes.flatten()[i].set_xlabel('')
            axes.flatten()[i].set_ylabel('')
            axes.flatten()[i].set_title(var, fontsize=22)
            axes.flatten()[i].legend(fontsize=20)
            axes.flatten()[i].tick_params(axis='both', labelsize=22)
            fig.text(0.5, 0.07, 'Paths per prediction', ha='center', fontsize=32)
            fig.text(0.05, 0.5, r'Predicted log($k_{cat}$)', va='center', rotation='vertical', fontsize=32)

        # fig.tight_layout()
        if figname:
            fig.savefig(figname, dpi=300)
        
        return results
    
    def corr_test(self, verbose=False):
        # Calculate Pearson and Spearman correlation coefficients between predictions and 
        # TIS measured kcat values for variants in the test set
        
        # From https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html:
        # Returns:
        # res -- a dictionary containing keys and values:
        #        'spearman': SignificanceResult returned by scipy.stats.spearmanr
        #        'pearson': PearsonRResult returned by scripy.stats.pearsonr
        #    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html
        #        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html
        #    for more information on these returned objects
        

        # Populate the obs attribute
        self.make_observations()

        variants, preds, targets = [], [], []
        # grab indexes of the held-out variants
        for i,var in enumerate(np.unique(self.test_vars)):
            res = self.score_var(var)
            res = np.concatenate(res)

            variants.append(var)
            preds.append(np.mean(res))
            targets.append(np.log10(self.obs.kcat[np.where(self.obs.variant==var)[0][0]]))

            print (f'Completed {i+1}/{np.unique(self.test_vars).shape[0]} variants...')

        if verbose:
            for i,var in enumerate(variants):
                print (f'{var}: {preds[i]:.2f}   |   {targets[i]:.2f}')
            
            plt.figure()
            plt.scatter(preds, targets)
            plt.scatter(-16.02, -16.02, marker='x', color='k', label='WT')
            plt.plot(targets, targets, ls='--', color='gray', label=r'$x=y$')
            plt.xlabel('Predictions')
            plt.ylabel('Targets (TIS)')
            plt.title(self.model_file)
            plt.legend()
            plt.show()
        
        res = {}
        res['spearman'] = spearmanr(preds, targets)#, alternative='greater')
        res['pearson']  = pearsonr(preds, targets)

        return res
    
    def corr_train(self, verbose=False):
        # Calculate Pearson and Spearman correlation coefficients between predictions and 
        # TIS measured kcat values for variants in the train set
        
        # From https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html:
        # Returns:
        # res -- a dictionary containing keys and values:
        #        'spearman': SignificanceResult returned by scipy.stats.spearmanr
        #        'pearson': PearsonRResult returned by scripy.stats.pearsonr
        #    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html
        #        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html
        #    for more information on these returned objects
        

        # Populate the obs attribute
        self.make_observations()

        variants, preds, targets = [], [], []
        # grab indexes of the training variants
        for i,var in enumerate(np.unique(self.train_vars)):
            res = self.score_var(var)
            res = np.concatenate(res)

            variants.append(var)
            preds.append(np.mean(res))
            targets.append(np.log10(self.obs.kcat[np.where(self.obs.variant==var)[0][0]]))

            print (f'Completed {i+1}/{np.unique(self.train_vars).shape[0]} variants...')

        if verbose:
            for i,var in enumerate(variants):
                print (f'{var}: {preds[i]:.2f}   |   {targets[i]:.2f}')
        
        res = {}
        res['spearman'] = spearmanr(preds, targets)#, alternative='greater')
        res['pearson']  = pearsonr(preds, targets)

        return res
    
    def tabulate_preds(self):
        # Calculates a prediction for each variant in the train and test set as the average
        # across all predictions made for that variant's observations of dynamics
        
        # Populate the obs attribute
        self.make_observations()

        self.variants = {'train':[],'test':[]}
        self.preds    = {'train':[],'test':[]}
        self.targets  = {'train':[],'test':[]}
        
        # Make predictions for the training variants
        for i,var in enumerate(np.unique(self.train_vars)):
            res = self.score_var(var)
            res = np.concatenate(res)
            self.variants['train'].append(var)
            self.preds['train'].append(np.mean(res))
            self.targets['train'].append(np.log10(self.obs.kcat[np.where(self.obs.variant==var)[0][0]]))
            
            if (i+1)%5 == 0:
                print (f'Completed {i+1}/{np.unique(self.train_vars).shape[0]} train variants...')
                
        # Make predictions for the test variants
        for i,var in enumerate(np.unique(self.test_vars)):
            res = self.score_var(var)
            res = np.concatenate(res)
            self.variants['test'].append(var)
            self.preds['test'].append(np.mean(res))
            self.targets['test'].append(np.log10(self.obs.kcat[np.where(self.obs.variant==var)[0][0]]))
            
            if (i+1)%5 == 0:
                print (f'Completed {i+1}/{np.unique(self.test_vars).shape[0]} test variants...')
        
        res = {'Spearman (test)':None, 'Spearman (train)':None, 'Pearson (test)':None, 'Pearson (train)':None}
        res['Spearman (test)']  = spearmanr(self.preds['test'],  self.targets['test'])#, alternative='greater')
        res['Spearman (train)'] = spearmanr(self.preds['train'], self.targets['train'])#, alternative='greater')
        res['Pearson (test)']   = pearsonr(self.preds['test'],   self.targets['test'])
        res['Pearson (train)']  = pearsonr(self.preds['train'],  self.targets['train'])
        
        return res
    
    def plot_preds(self, figsize=None):
        # Plots predictions vs. labels for both train and test data
        
        if self.variants == None:
            self.tabulate_preds()
            
        low = np.min(self.targets['train']+self.targets['test'])
        high = np.max(self.targets['train']+self.targets['test'])
        
        fig, ax = plt.subplots(1,1,figsize=figsize)
        ax.scatter(self.preds['train'], self.targets['train'], color='k', label='Train')
        ax.scatter(self.preds['test'], self.targets['test'], color='lightgreen', label='Test')
        # ax.scatter(-16.02, -16.02, marker='x', color='b', label='WT')
        ax.plot([low,high], [low,high], color='gray', label=r'$x=y$')
        ax.set_xlabel(r'Predicted $\mathrm{log}(k_{cat})$')
        ax.set_ylabel(r'Target $\mathrm{log}(k_{cat})$ (from TIS)')
        ax.set_title(self.model_file)
        ax.legend()
        
        fig
            
        return fig, ax
        
    @staticmethod
    def get_dir(file):
        # Returns direction under which file is stored
        file_dir = '/'.join(file.split('/')[0:-1]) + '/'
        return file_dir

    @staticmethod
    def get_val_variants(model_file, output_file=None):
        # Returns list of variants that were held out
        # during training of the model saved to model_file
        # INPUT:
        # model_file -- full path to the file to which the 
        #               model was saved
        # output_file -- the text file to which output was 
        #                written by the script that trained
        #                and saved the model

        if output_file == None:
            output_file = ModelTest.get_dir(model_file) + 'transformer_1_output.txt'

        if ModelTest.get_dir(model_file) != ModelTest.get_dir(output_file):
            raise ValueError('model_file and output_file directories do not match.' +\
                             'Are you sure you have the right ones?')

        # get the CV fold
        fold = model_file.split('cvfold')[-1].split('.pt')[0]

        # read the output file
        with open(output_file, 'r') as f:
            data = f.read()

        # grab the variants used for training
        train_vars = data.split(f"epoch 1 | CV fold {fold}")[0].split('Training variants:')[-1]

        # grab the variants used for validation
        val_vars = data.split(f"Best loss achieved, saving model state to best_model_cvfold{fold}.pt")[0]
        val_vars = val_vars.split('Validation variants:')[-1]
        
        train_vars = train_vars.replace("' '",',').replace("'",'').replace('\n ',',').split('[')[-1].split(']')[0].split(',')
        val_vars = val_vars.replace("' '",',').replace("'",'').replace('\n ',',').split('[')[-1].split(']')[0].split(',')

        return np.array(train_vars), np.array(val_vars)