"""
AUTHOR: E Karvelis (karvels2@mit.edu)
DATE:   April 25, 2023

PURPOSE: 
Predict TIS-calculated k_cat values for different enzyme mutants as a 
function of their active site dynamics leading up to attempted turnover
events. Observations of the dynamics leading to, and sometimes including,
attempted turnover are referred to as trajectories or pathways. The 
goal of this script is to train a transformer-based model that predicts 
the k_cat for a given mutant based on a sampled set of its trajectories.

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
cv_folds --         The number of cross validation folds to use. Default, 5. (int)
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
epochs --           Number of training epochs. Default, 100. (int)
d_model -- 			The dimensions of the transformer encoder layers in the, or hidden 
					layer sizes inside transformer encoder. All sublayers in the model 
					will produce outputs with this dimension. (int)
warmup_steps --     The number learning rate warmup steps to use during model training.
                    The learning rate is gradually and linearly increased over the 
                    course of warmup_steps training steps (each training batch and 
                    update of learnable parameters is considered a single step) before 
                    it decays over the course of further training. See function 
                    warmup_decay_lr() or the Attention is All You Need paper (), on 
                    which it is based, for more information. (int)

d_model -- 			The dimensions of the transformer encoder layers in the 
           			transformer encoder. All sublayers in the model will produce
           			outputs with this dimension. (int)
d_input_enc --      Hidden layer size for the (middle layer of the) 2 layer input encoder.
					Default, 128. (int)
n_head -- 			The number of attention heads (parallel attention layers) in 
          			each transformer encoder layer. Default, 8. (int)
d_tran_ffn -- 		Number of neurons in the linear feedforward layer of the transformer
              		encoder layers. Default, 2048. (int)
dropout_tran_ecoder -- Dropout for the transformer encoder layers. Default, 0.2.
                       (float)
n_tran_layers -- 	Number of stacked transformer encoder layers in the transformer
                 	encoder. Default, 4. (int)
d_mlp_head -- 		Hidden layer size for the (middle layer) of the two layer MLP
              		regression head. (int)
dropout_mlp_head -- Dropout rate applied in between the two layers in the 
                    MLP regression head. (float)
control_model --    Whether to train a negative control model or not. If True, then
					the target (labels) will be scrambled for the training data 
					prior to model training (validation sets' labels are not 
					scrambled). This setting trains a negative control model to 
					get a sense of baseline performance for a naive model that 
					guesses the mean across all variants, for every variant. If
					False, then no data are scrambled. Default, False. (bool)
stoch_labels --		Whether to stochastically sample labels (i.e, log(kcat) values)
					according to the mean and standard error of the mean (SEM) for
					TIS-calculated kcat values, which are specific to each mutant.
					Each variant will have multiple 'observations,' or sets of paths
					from which a prediction is made. When stoch_labels is False, the
					labels for these observations will always be the mean kcat. When 
					stoch, labels is True, the label for each observations will be 
					sampled from a normal disribution with mean = mean kcat and 
					standard deviation = SEM kcat. Default, False. (bool)
selected_variants -- The variants whose data to use. This setting allows one to 
					 specify that only a subset of the variants are to be used. 
					 For example, if you wish to train and test on only WT, you
					 could pass selected_variants = ['WT']. Or if you wanted 
					 to train on only a few of the fastest mutants, you could 
					 pass ['GLn140Met-Thr520Asp', 'Leu501His','Thr520Asp']. 
					 Defaults to ['*'], in which case all variants available in 
					 the dataset are used. (list of strings, or the character '*')
features --         Names of the structural features to use for model training and 
                    evaluation. Default, corresponding to features = ['*'], is to 
					use all available features (70 different interatomic distances, 
					angles, and dihedral angles), but by specifying this variable, 
					you can select a subset of features. You must specify features 
					that are actually in the dataset by name. For example, 
					features = ['Dist AC6/C5,AC6/C4', 'Dist AC6/C5,AC6/C7']
					The order doesn't matter; it won't affect the data at all.
					You can access a list of feature names by loading meta_file 
					(see above) to some object, say meta, and then printing 
					meta.ft_names.  
					Default, ['*']. (list of strings)
task 			 -- The type of learning task for which to train and evaluate the 
					model. Must be one of:
					'kcat regression' -- regression task to predict log10(kcat)
					'NR/R binary classification' -- classify paths as either non-reactive or
											 		reactive
					'S/F binary classification' -- classify paths as belonging to a mutant
											 	   that was either slower or faster than WT
					Default, 'kcat regression'. (str)
dw_settings 	 -- Passes the parameters to use inside the DenseWeight method for
					controlling the weighting of individual sample's terms in the loss
					function. The DenseWeight method is described here:
					https://link.springer.com/article/10.1007/s10994-021-06023-5, 
					and implemented using their Python package here:
					https://github.com/SteiMi/denseweight, with inspiration taken
					from the original paper's GitHub repo here:
					https://github.com/SteiMi/density-based-weighting-for-imbalanced-regression/tree/main/exp1_and_2.
					dw_settings is a dictionary specifying kwargs for DenseWeight, e.g.
					
					dw_settings = {'alpha':0.9, 'bandwidth':1, 'eps':1e-6'}
					
					'alpha' controls the strength of the weighting. When set 
						to 0, no weighting is applied and all terms are 
						weighed equally. 'alpha' controls the alpha parameter as
						described in the paper. Increasing its value places relatively
						larger weights on the loss terms corresponding to less 
						frequent samples/observations. Therefore, set alpha to
						positive values (0.0 - 2.0 is probably a good range) to 
						more strongly weight less frequent values as a means to give 
						more equal influence across the range of kcat values in 
						our dataset (which is largely biased toward kcat near WT 
						value, otherwise). The higher the value of alpha, the 
						more the weighting emphasizes differences in density of 
						samples/observations. Values around 0.9 are resonable. (float)
					'bandwidth' controls the bandwidth of the kernel density-fitting
					    function used by DenseWeight to calculate the empirical 
					    probability density. Smaller bandwidth = higher resolution.
					    Larger bandwidth = smoother density. A value of 1, the method's
					    default, is reasonable. (float)
					'eps' sets the minimum weight to be applied to a term in the 
					    loss function. The DensWeight default is 1e-6. (float)

					Default, dw_settings = None, in which case no loss weighting 
					is applied. (dict)				
parallel_folds   -- Whether to train the different CV folds in parallel (True) or
					sequentially (False). Generally, only set to True when submitting
					to a cluster. It is suggested that when True, you submit the 
					job to 2*cv_folds number of processors. Default, False. (bool)
num_processes    -- The number of folds to be run in parallel. Only used when 
					parallel_folds = True, otherwise ignored. Default, 3. (int)
mixed_variants   -- Whether to pool together pathways from different variants in each
                    observation (an observation is a set of pathways that are passed
					together as input to the model. One observation is paired with 
					one output label). If doing a classification task, then paths 
					will be pooled from only those variants with the same output
					label. If doing a regression, then paths will be pooled according
					to probabilities that scale with the similarity between their 
					labels. True, paths are mixed across different variants. False,
					paths are not mixed, and all observations of comprised of data 
					from a single variant. Default, False. (bool)
regularization   -- Specifies which type of regularization to apply to the control the
					overall magnitudes of trained weights. Options are None, L1, or L2 
					regularization. Default, None, in which case no penalization term is
					added. Otherwise, specifies a dictionary whose key is either 'l1'
					or 'l2' and value is the lambda coefficient to scale the relative 
					size of the penalization term (suggested values: 0.01, 0.001). E.g.,
					regularization = {'l1', 1e-5}.
					Default, None. (dict)
time_range --       Range of time points to include from data. If set to ['*'], then
 					all the time points available in the dataset will be used. 
					Otherwise, time specifies the minimum and maximum time points 
 					defining the time window (inclusive). E.g., time = [-160,-130]
					specifies a 31 step-long time window from -160 fs to -130 fs.
					Default, ['*']. (list of str or int)

"""

""" ============================================== """
###                     Packages                   ###
""" ============================================== """
import torch.multiprocessing as mp 
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, KFold, StratifiedKFold
import pandas as pd
from denseweight import DenseWeight
import matplotlib.pyplot as plt
""" ============================================== """


""" ============================================== """
###                     Main code                  ###
""" ============================================== """
if __name__ == '__main__':


	""" ============================================== """
	###       Load settings and set up environment     ###
	""" ============================================== """
	import sys
	import time
	mp.set_start_method('spawn')

	# read in settings to overwrite defaults
	split_by_variant = True
	control_model = False
	random_seed = 333
	path_set_size = 10
	batch_size = 32
	cv_folds = 5
	epochs = 50
	output_text = './transformer_1_output.txt'
	d_model = 256
	d_input_enc = 128
	n_head = 4
	d_tran_ffn = 1024
	dropout_tran_encoder = 0.2
	n_tran_layers = 2
	d_mlp_head = 128
	dropout_mlp_head = 0.2
	warmup_steps = 4000
	stoch_labels = False
	selected_variants = ['*']
	features = ['*']
	task = 'kcat regression'
	dw_settings = None
	parallel_folds = False
	num_processes = 3
	mixed_variants = False
	regularization = None
	time_range = ['*']

	# read in settings to overwrite defaults
	with open(sys.argv[1], 'r') as f:
		settings = f.read()
		exec(settings)

	# import dependencies from prep_data.py
	sys.path.append(f'{loc}/scripts/')
	from prep_data import *
	from pred_kcat import *

	# initialize random number generator
	np.random.seed(random_seed)

	# open a file for writing progress
	output_text_filename = output_text
	output_text = open(output_text, 'w')

	# device configuration
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print ('DEVICE: ', device)

	# model parameters
	if features[0] == '*':
		input_size =  int(data_file.split('-')[-1].split('memnpy')[0]) # number of features
	else:
		input_size = len(features)
	if time_range[0] == '*':
		input_length = int(data_file.split('.')[-1].split('-')[1])
	else:
		input_length = int(time_range[1] - time_range[0] + 1)

	model_kwargs = {'input_size': input_size,
					'input_length': input_length,
					'd_model': d_model,
					'd_input_enc': d_input_enc,
					'n_head': n_head,
					'd_tran_ffn': d_tran_ffn,
					'dropout_tran_encoder': dropout_tran_encoder,
					'n_tran_layers': n_tran_layers,
					'd_mlp_head': d_mlp_head,
					'dropout_mlp_head': dropout_mlp_head,
					'task': task}

    # explicitly list the variants, if using all of them; convert to array
	tmp_meta = pickle.load(open(meta_file, 'rb'))
	if selected_variants[0] == '*':
		# then use all the variants
		selected_variants = np.unique(tmp_meta.variant)
	else:
		selected_variants = np.array(selected_variants)
		
    # list the variants' associated kcat values        
	tmp_kcat = [lookup_kcat(v, tmp_meta)[0] for v in selected_variants]

	# define CV splitter
	if split_by_variant and (selected_variants.shape[0] >= cv_folds):
		groups = selected_variants
		if task == 'S/F binary classification':
			cv = StratifiedGroupKFold(n_splits=cv_folds)
			y = np.array(np.log10(np.array(tmp_kcat)) > -16.02, dtype=int)
			splits = cv.split(selected_variants, y=y, groups=groups)
		else:
			cv = GroupKFold(n_splits=cv_folds)
			splits = cv.split(selected_variants, groups=groups)

	else:
		data = PathDataset(data_file, meta_file, path_set_size=path_set_size,
				selected_variants=selected_variants, task=task, features=features, 
				mixed_variants=mixed_variants, time_range=time_range)
		data.make_observations()
		groups = np.random.choice(cv_folds, size=data.obs.obs.shape[0])
		if task == 'S/F binary classification':
			cv = StratifiedGroupKFold(n_splits=cv_folds)
			y = np.array(np.log10(np.array(data.obs.kcat)) > -16.02, dtype=int)
			splits = cv.split(np.arange(data.obs.obs.shape[0]), y=y, groups=groups)
		else:
			cv = GroupKFold(n_splits=cv_folds)
			splits = cv.split(np.arange(data.obs.obs.shape[0]), groups=groups)

	# define multiprocessing pool, if needed
	if parallel_folds:
		pool = mp.Pool(processes=num_processes)
		processes = []

	tot_start_time = time.time()



	""" ============================================== """
	###                     CV loop                    ###
	""" ============================================== """
	for i, (trainset, testset) in enumerate(splits):
		# trainset and testset each hold a list of indexes to split selected_variants
        # into a train and test set of variants if select_by_variant
        # otherwise, they each hold a list of indexes to split observations from data
        # into train and test sets
		start_time = time.time()
		cv_fold = i+1

		if split_by_variant:
			traindata = PathDataset(data_file, meta_file, path_set_size=path_set_size,
					    selected_variants=list(selected_variants[trainset]), task=task, 
						features=features, mixed_variants=mixed_variants, time_range=time_range)
			testdata =  PathDataset(data_file, meta_file, path_set_size=path_set_size,
                        selected_variants=list(selected_variants[testset]), task=task, 
						features=features, mixed_variants=False, time_range=time_range)
			traindata.make_observations()
			testdata.make_observations()
			
			fold = Fold(cv_fold, traindata, testdata, parallel_folds,
                        output_text_filename, model_kwargs, device,
                        stoch_labels, dw_settings, control_model, epochs,
                        random_seed, batch_size, cv_folds, model_type='transformer encoder',
						train_idxs=None, test_idxs=None, regularization=regularization)
			
		else:
			fold = Fold(cv_fold, data, data, parallel_folds,
                        output_text_filename, model_kwargs, device,
                        stoch_labels, dw_settings, control_model, epochs,
                        random_seed, batch_size, cv_folds, model_type='transformer encoder',
						train_idxs=trainset, test_idxs=testset, regularization=regularization)

		if parallel_folds:
			processes.append(pool.apply_async(fold.run, ()))
		else:
			fold.run()

		print (f"Fold {i+1} runtime: {time.time() - start_time}")


	if parallel_folds:
		# Run the folds
		results = [p.get() for p in processes]

		# Copy all folds' output to main output file
		fold_files = glob(f'{output_text_filename}-fold*')
		folds = [int(x.split('-fold')[-1]) for x in fold_files]
		fold_files = [x for _,x in sorted(zip(folds, fold_files))]
		
		for fold_file in fold_files:
			f = open(fold_file, 'r')
			output_text.write(f.read())
			output_text.flush()
			f.close()
			os.remove(fold_file)
		
		output_text.close()

	else:
		output_text.write(f'Total runtime: {time.time() - tot_start_time}\n')
		output_text.flush()
		output_text.close()
		
	print ('DONE')
	print (f'Total runtime: {time.time() - tot_start_time}')






