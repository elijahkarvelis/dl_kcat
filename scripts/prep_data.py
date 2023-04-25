"""
Authors: E Karvelis
Date: 3/17/2023

Purpose:

To run as main, run like so:

>> python prep_data.py config.txt

where config.txt file contains the settings for the script.
For example, one config.txt file might read:


# Specification of transition pathway data:
tp_data = True
order_parameters = [-0.4,0.8]
seeds = '*'
replicates = '*'
time = [-110,0]
features = None
num_shots = None
subsample = 500

# Specifcation of whether to load GS (equilibrium) data:
gs_data = True

# Global specification of which variants to include
variants = ['Gln140Met-Thr520Asp','Leu501His','Met472Lys']
variants_summary = './input/variants_summary.csv'

# Set random seed
random_seed = 333

# Specify where to load data (if available) or to save (if needed to compile data)
load_file     = './data/o-0dot4_0dot8_sall_rall_t-110_0_sub300_numNone.data'
save_filename = './data/o-0dot4_0dot8_sall_rall_t-110_0_sub300_numNone.data'

# Specify the location of the package (the local git repo)
package_loc = '/data/karvelis03/dl_kcat/'


INPUTS:
tp_data -- True or False. Whether to load and include data from transiton pathways. Default, True
gs_data -- True or False. Whether to laod and include data from equlibrium simulations (GS). Default, True
order_parameters -- This is a str representation of a python list that lists the
                    order parameters whose ensembles data should be included in
                    training/predicting enzyme activity from transition pathways.
                    Defaults to [-0.4, 0.0]. See modes 1 and 3, or the
                    perform_iterative_test() fxn. for more info.
seeds -- the numbers of the seed pathways whose ensemble data under the
         ./tis_mldata/ folder you want to include in the analysis. Can be
         either a single integer (to use just one seed pathway's data), or a
         list of integers (to use multiple seed pathways' data). To use data
         from all available seed pathways, you can use * , which is shorthand
         for grabbing all available data. Using all available data is also the
         default setting. (int, list, or str)
replicates -- the numbers of the ensemble replicates whose data under the
              ./tis_mldata/ folder you want to include in the analysis. Can be
              either a single integer (to use just one replicate's set of data
              for each seed requested), or a list of integers (to use
              multiple replicates' data sets for each seed requested). To use
              data from all available replicates, you can use * , which is
              shorthand for grabbing all available data. Using all available
              data is also the default setting. (int, list, or str)
time -- A range of time points between which to average, or randomly sample,
        the features for each trajectory. For example, time = [-200,100]
        will grab the 200 time points preceding t = 0 through the 100 time
        points following t = 0. To use features at a specific, single time point,
        as opposed to taking the average over some time interval, you can write
        the time point of interest twice. For example, time = [-100,-100] will
        grab the data taken at t = -100 fs (100 frames or time steps before t = 0)
        and only at t = -100 fs (no averaging is done!). This variable defaults to
        t = [-100,-50], i.e. overaging between t = -100 fs and t = -50 fs. (list)
pooled -- whether to randomly sample, and pool together, different time points'
          frames' features for use as input data. DEFAULTS TO FALSE, which means
          that each pathway's features will be averaged across the time points
          listed in time, as opposed to pooling together random time points.
          If not false, but instead an integer say n, then n random time points
          will be drawn from frames between the two time points listed in time.
          So, this variable specifies how many frames to sample from each pathway.
          If pooled = "true", then it defaults to sampling 1 random time point
          from each pathway. (str for "false", which gets converted to boolean;
          or integer).
features -- what features to include for the classifier to use (str of list).
            Defaults to None, in which case all are used.
num_shots -- the number (int) of shots to include from each ensemble. If
             left blank, then defaults to None and uses all available shots.
             Usually this is left blank because we want to use all the available
             data. But, if you want to specifically test how many shots/ensemble
             are required for good model performance, this parameter can be employed.
subsample -- the number (int) of pathways to randomly sample from each ensemble
             (weighted by pathway counts in the ensemble) to include in analysis.
             Working with a subset of data speeds up training time, and also 
             allows us to test how much data are required for training. Defaults to 
             None, in which case all data are used (no subsampling)
variants -- list of which variants to use. Variants' names must match format
            used in variants_summary file. To use all variants with TIS kcat
            estimates, set to '*'.
variants_summary -- name of the CSV file storing summarized data on all variants.
                    This CSV file is provided in the repo's ./input/ subfolder 
                    as ./input/variants_summary.csv.
random_seed -- number (int) to initialize random number generator
load_file -- name of file to try to load data from. If the file doesn't exist, 
             then the data will be compiled and saved to a file with name 
             save_filename. Generally, load_file == save_filename.
             Defaults to None, in which a descriptive name will be automatically
             assigned, something like './data/o-0.4_0.8_sall_rall_t-110_0_sub300_numNone.data'
save_filename -- if the data are newly compiled, then they will be saved to 
                 save_filename. Generally, save_filename == load_file. The 
                 purpose of load_file and save_filename is to avoid having 
                 to re-compile the same dataset over and over, when we could
                 simply load it from file after the first time it's generated.
                 Defaults to None, in which a descriptive name will be automatically
                 assigned, something like './data/o-0.4_0.8_sall_rall_t-110_0_sub300_numNone.data'
package_loc -- (str) path to the package; this is the local location of the repo. For me, it's 
               '/data/karvelis03/dl_kcat/'. Defaults to './', in which case it is assumed the working 
               directory is the repo. It is very important to specify this parameter correctly, because 
               the working scripts have several dependencies on the files under the repo's './input/'
               subfolder.

"""

"""
Set up: load packages and set seed
"""
import sys
import os
from glob import glob

# import packages for numerical data
import numpy as np
import pandas as pd
import pickle

"""
Class that defines and object to represent data from a tis.py-written output_text.txt file
"""
class Output:
    # This is a python object that represents E Karvelis TIS scripts' output_text.txt files

    def __init__(self, filename):
        self.filename = filename      # name of output_text.txt file
        self.ids = []                 # list of accepted shooting attempt numbers in order
        self.counts = []              # list of accepted shooting attempts' counts in order
        self.traces = []              # list of accepted shooting attempts' traces in order
        self.crossings = []           # binary list of whether shooting attempts crossed lam_(i+1). 1=yes; 0=no
        self.num_shots = None         # number of shooting attempts (size of the ensemble)
        self.acc_shots = None         # number of accepted attempts (# of unique paths in ensemble)
        self.cross_shots = None       # number of unique, accepted shots that crossed lam_(i+1)
        self.p_term = None            # the P_term for interface, given by: cross_shots*(their counts) / num_shots

        self.all_traces = []          # for debugging purposes ONLY; all traces in trajsum.txt, some are repeated

    def info(self):
        print ("self.filename = filename      # name of output_text.txt file")
        print ("self.ids = []                 # list of accepted shooting attempt numbers in order")
        print ("self.counts = []              # list of accepted shooting attempts' counts in order")
        print ("self.traces = []              # list of accepted shooting attempts' traces in order")
        print ("self.crossings = []           # binary list of whether shooting attempts crossed lam_(i+1). 1=yes; 0=no")
        print ("self.num_shots = None         # number of shooting attempts (size of the ensemble)")
        print ("self.acc_shots = None         # number of accepted attempts (# of unique paths in ensemble)")
        print ("self.cross_shots = None       # number of unique, accepted shots that crossed lam_(i+1)")
        print ("self.p_term = None            # the P_term for interface, given by: cross_shots*(their counts) / num_shots")
        print ("self.all_traces = []          # for debugging purposes ONLY; all traces in trajsum.txt, some are repeated")
       # print ("self.dcds = []                # names of all stitched DCD files in order; populate w/ self.load_stitched_dcds()")
       # print ("self.crd = []                 # names of the standard CRD file; populated by self.load_stitched_dcds()")
        print ("\nAttributes populated by running self.set_up()")

    def set_up(self):
        # this function reads the trajsum.txt file and populates the object's attributes
        with open(self.filename, 'r') as f:
            attempts = [x for x in f.read().split('Shot ID:')][1::]

        prev_p_num = 0
        count = 0
        for attempt in attempts:
            ID = int(attempt.rsplit()[0].replace('-pad',''))
            p_den = int(attempt.split('P_denominator')[-1].rsplit()[2])
            p_num = int(attempt.split('P_numerator')[-1].rsplit()[3])
            tot_acc = int(attempt.split('tot_acceptances')[-1].rsplit()[4])
            trace = attempt.split('step:')[-1].split(']')[0][2::].rsplit()
            trace = np.array(trace, dtype=float)
            if not (ID in self.ids):
                self.all_traces.append(trace)

            if 'FAILED' in attempt:
                # Rejected
                count += 1 # this count variable is tracking the previous accepted attempt
                if ID == 1: # this is a special case; gets handled differently
                    self.ids.append(0)
                    self.traces.append(trace)
                    self.crossings.append( int(p_num > prev_p_num) )
            else:
                # Accepted
                ### EDIT EK 1/22/21: check if ID is in self.ids; if so, then you restarted the TIS simulation and already recorded it's info, so skip adding its data a second time so you don't over count
                if ID in self.ids:
                    None

                else:
                    self.ids.append(ID)
                    self.traces.append(trace)
                    self.crossings.append( int(p_num > prev_p_num) )
                    if ID != 1:
                        self.counts.append(count) # this count corresponds to the previous accepted attempt
                count = 1 # start count for this ID

            prev_p_num = p_num

        self.counts.append(count) # append the count of the last attempt that was accepted


        p_denominator = int(attempts[-1].split('P_denominator')[-1].rsplit()[2])
        p_numerator = int(attempts[-1].split('P_numerator')[-1].rsplit()[3])
        tot_acceptances = int(attempts[-1].split('tot_acceptances')[-1].rsplit()[4])

        # Compute useful values:
        self.num_shots = p_denominator
        self.acc_shots = tot_acceptances
        self.cross_shots = p_numerator
        self.p_term = p_numerator / p_denominator

        """
        # raise error if something is off:
        if not (len(self.traces) == len(self.counts)):
            raise ValueError("Numbers of traces and counts don't match")
        if not (len(self.ids) == len(self.counts)):
            raise ValueError("Numbers of ids and counts don't match")
        if not (len(self.crossings) == len(self.counts)):
            raise ValueError("Numbers of crossings indications and counts don't match")
        if not (self.acc_shots == len(self.counts)):
            raise ValueError("Number of accepted shots doesn't match size of data")
        if not (self.num_shots == np.sum(np.array(self.counts, dtype=int))):
            raise ValueError("Number of shots doesn't match sum of counts")
        """

"""
Class that defines an object to represent post-processed data from a pathway ensemble
"""
class Features:
    # this is a class with attributes:
    # filename -- the name of the features.npy file summarized by the object
    # data -- matrix of the form [pathways, timepoints, features]
    # ft_names -- array listing feature names in the same order as they are in self.data
    # ids -- array listing pathway ids in the same order as they are in self.data
    # counts -- array listing the counts (weights) of each pathway in order corresponding
    #                to how pathways are listed in self.data
    # t0 -- the index of the timepoint corresponding to t = 0

    def __init__(self, output_file, pathway_ids_filename, rev_l):
        # Defines:
        # self.filename
        # self.data
        # self.ft_names
        # self.ids
        # self.counts
        # self.t0

        save_dir = '/'.join(output_file.split('/')[0:-1]) + '/'
        output_text_filename = save_dir + 'output_text.txt'

        self.filename = output_file
        self.data = np.load(output_file)
        self.ids = np.load(pathway_ids_filename)
        self.ft_names = np.load(save_dir+'feature_names.npy')
        self.t0 = rev_l

        # get counts from trajsum file:
        output = Output(output_text_filename)
        output.set_up()
        output_ids = np.array(output.ids, dtype=int)
        counts = []
        for path_id in self.ids:
            count = output.counts[np.where(output_ids == int(path_id))[0][0]]
            counts.append(count)
        counts = np.array(counts, dtype=int)
        self.counts = counts



    def info(self):
        print ("self.filename = str          # name of features.npy file summarized by object")
        print ("self.data = array            # features data matrix of form: [pathways,timepoints,features]")
        print ("self.ft_names = array        # names of the features in order as they appear in self.data")
        print ("self.ids = array             # ids of pathways in order as they appear in self.data")
        print ("self.counts = array          # counts of pathways in order as they appear in self.data")
        print ("self.t0 = int                # the index fo the timepoint corresponding to t = 0")

"""
Helper function to find frame corresponding to t0 in a raw DCD file
"""
def find_t0(dcd, crd):
    # given a dcd filename (dcd) for which features have already been computed
    # and saved to dcd.replace(".dcd",".npy") or dcd.replace(".dcd","exp.npy"),
    # this tells you the name of frame where t=0 in the raw trajectory
    # so, if you watch the dcd in PyMOL, scroll to frame t0+1 to find where
    # t0 is in the movie

    # helper function
    def dist(atom1, atom2):
        # input: two atoms of interest (type: MDAnalysis atomgroup object)
        # returns: the distance between them

        r = atom1.position - atom2.position
        d = np.linalg.norm(r)

        return d

    # make a universe of your trajectory
    u = MDAnalysis.Universe(crd, dcd, format='DCD')
    AC6_C4 = u.select_atoms('segid C and name C4')[0]
    AC6_C5 = u.select_atoms('segid C and name C5')[0]
    AC6_C7 = u.select_atoms('segid C and name C7')[0]

    fts = {'break': [], 'make': []}
    for ts in u.trajectory:
        fts['break'].append(dist(AC6_C4, AC6_C5))
        fts['make'].append(dist(AC6_C5, AC6_C7))

    # find the index of the row where we define t=0:
    lam_traj = np.array(fts['break'], dtype=float) - np.array(fts['make'], dtype=float)
    dlam_traj = np.gradient(lam_traj) # derivative of lambda values over time
    t0 = np.argmin(np.abs(lam_traj)) # where lambda almost equal 0

    # now we scan backwards from t0 until d(lambda)/dt is < 0
    derivative = dlam_traj[t0]
    check = True
    while (derivative > 0) or check:
        t0 -= 1
        derivative = dlam_traj[t0]

        # check that negative gradient persists
        d1 = dlam_traj[t0-1]

        if (lam_traj[t0]<-0.3):
            check = False
        else:
            check = True

    return t0

"""
Loads objects of a given order parameter
"""
def load_objs(order, seeds='*', replicates='*', loc="./", num_shots=None, file_prefix="", subsample=None):
    """
    Inputs:
    order - the order parameter of the objects to load (float)
    seeds - the numbers of the seeds of the objects to load (defaults to loading all seeds'). String, int, or list of strings and/or integers
    replicates - the numbers of the replicates of the objects to load (defaults to loading all replicates). String, int, or list of strings and/or integers
    loc - the path to the directory containing the ./tis_mldata/ folder containing the data on which to run this script's analysis
    num_shots - the number of shots to include from each ensemble. Int or None. None, the default,
                will use all available shots

    Outputs:
    objs - a list of the loaded feature_set objects of the given order
    """

    # convert seeds and replicates to correct data types:
    if type(seeds) == int:
        seeds = [str(seeds)]
    else:
        seeds = list(map(str, list(seeds)))
    if type(replicates) == int:
        replicates = [str(replicates)]
    else:
        replicates = list(map(str, list(replicates)))

    # Instantiate a list to hold the feature objects
    objs = []
    if order > 0.5:
        # Set the prefix to indicate reacting
        prefix = loc+f'tis_mldata/seed*/reactive_r*/output/{file_prefix}features.obj'
    else:
        # Set the index to indicate not reacting
        prefix = loc+f'tis_mldata/seed*/almost_reactive_lam%0.1f_r*/output/{file_prefix}features.obj' %order

    fs = glob(prefix)

    if seeds[0] != '*':
        # filter for the desired seeds
        fs = [f for f in fs if f.split('mldata/seed')[-1].split('/')[0] in seeds]

    if replicates[0] != '*':
        # filter for the desired replicates
        fs = [f for f in fs if f.split(f'/output/{file_prefix}features.obj')[0].split('_r')[-1] in replicates]

    fs.sort()

    for f in fs:
        with open(f, 'rb') as fo:
            obj = pickle.load(fo)

            if num_shots != None:
                # sort all data in obj by ids
                sorter = np.array(obj.ids, dtype=int)
                sorting_indexes = np.argsort(sorter)
                obj.ids = obj.ids[sorting_indexes]
                obj.counts = obj.counts[sorting_indexes]
                obj.data = obj.data[sorting_indexes, :, :]
                # cut down obj data to only include data from shots 0 though num_shots
                max_id = np.where(np.cumsum(obj.counts) >= num_shots)[0][0] + 1
                obj.data = obj.data[0:max_id, :, :]
                obj.ids = obj.ids[0:max_id]
                obj.counts = obj.counts[0:max_id]
                # adjust last element in obj.counts to new end point
                #obj.counts[-1] = num_shots - int(obj.ids[-1]) + 1
                # ^^ doesn't work for NR ensembles because we artifically
                #    remove accidentally reactive paths
                obj.counts[-1] = num_shots - np.sum(obj.counts[0:-1])

            elif subsample != None:
                # randomly sample indexes to keep
                rand_selection = np.random.choice(np.arange(obj.data.shape[0]), size=subsample, replace=False, p=obj.counts/np.sum(obj.counts))
                obj.data = obj.data[rand_selection, :, :]
                obj.ids = obj.ids[rand_selection]
                obj.counts = obj.counts[rand_selection]


            objs.append(obj)

    return objs

"""
Helper function that handles conversion between t0-relative time points and acutal indexes
"""
def make_trange(obj, trange=None):
    # pass it a Features object (obj) and a time point range (trange) as a list.
    # trange defaults to entire time span
    # For example, trange = [-200,100] will grab the 200 time points
    # preceding t = 0 through the 100 time points following t = 0
    # RETURNS: plot_trange - array of indexes corresponding to the desired time points
    #          timesteps - array of time points corresponding to the indexes

    if trange == None:
        plot_trange = [-obj.t0,obj.data.shape[1]-obj.t0-1]
    else:
        plot_trange = trange

    plot_trange = [i+obj.t0 for i in plot_trange]
    plot_trange = np.arange(plot_trange[0],(plot_trange[1]+1))
    if np.sum(plot_trange < 0) > 0:
        raise ValueError("Lower bound time step given in trange is outside the time interval recorded for obj")

    timesteps = plot_trange-obj.t0

    return plot_trange, timesteps

"""
Averages data points over a time range
"""
def average_over_time_points(obj, time=None):
    """
    Inputs:
    obj - A loaded feature_set object (Features object)
    time - A range of time points between which to average the features for each trajectory (list)
           For example, time = [-200,100] will grab the 200 time points
           preceding t = 0 through the 100 time points following t = 0
           Default (None) uses all available time points
    Outputs:
    obj - the input object with time averaged data as an added property (Features object)
    """

    time,_ = make_trange(obj, trange=time)

    # Average each trajectory over the time indexes
    obj.averaged_data = np.mean(obj.data[:,time,:], 1)

    return obj

"""
Samples data points between a time range
"""
def sample_between_time_points(obj, pooled=5, time=None, series=False):
    """
    Inputs:
    obj - A loaded feature_set object (Features object)
    time - A range of time points between which to sample the features for each trajectory (list)
           For example, time = [-200,100] will grab the 200 time points
           preceding t = 0 through the 100 time points following t = 0
           Default (None) uses all available time points
    pooled - the number of time points to randomly sample (bewteen the time points
             in time) for each pathway in the obj
    series - this indicates that the time window should be loaded in as a series or sequence 
             of time points (not randomly sampled or shuffled) when True. When False (default), 
             time points will be randomly sampled and ordered. Should set series == True when
             compiling data for sequential models such as RNNs.

    Outputs:
    obj - the input object with time averaged data as an added property (Features object)
    """

    time,time_labels = make_trange(obj, trange=time)

    sampled_time, sampled_time_labels = [], []
    for i in range(obj.data.shape[0]):
        if series:
            selections = np.arange(time.shape[0])
        else:
            selections = np.random.choice(np.arange(time.shape[0]), size=pooled, replace=False)
        sampled_time.append(time[selections])
        sampled_time_labels.append(time_labels[selections])

    sampled_time = np.array(sampled_time, dtype=int)
    sampled_time_labels = np.array(sampled_time_labels, dtype=int)

    # sub-sample each trajectory, keeping only the selected time indexes
    sampled_time = sampled_time[:,:,np.newaxis] # adds third axis
    obj.averaged_data = np.take_along_axis(obj.data,sampled_time,1)
    obj.sampled_time_labels = sampled_time_labels

    # reshape the arrays, such that each frame is given its own row; this now
    # means that a single pathway has data in multiple rows of obj.averaged_data
    if not series:
        obj.averaged_data = np.reshape(obj.averaged_data, (-1,obj.averaged_data.shape[-1]))
        obj.sampled_time_labels = np.reshape(obj.sampled_time_labels, (-1,1))

    return obj

"""
Adds the trajectory counts as a feature to the averaged data
"""
def add_counts_and_ids_and_seeds_and_replicates_as_features(obj, pooled=False):
    """
    Inputs:
    obj - A loaded and time averaged Features object
    Outputs:
    data - the time-averaged data with trajectory counts, IDs, and corresponding
           starting seed and statistical replicate numbers appended
           (matrix with trajectories as rows and features as columns)
    """

    # Get the averaged data
    data = obj.averaged_data
    # Get the count for each trajectory
    counts = obj.counts
    # Get the pathway ID for each trajectory
    ids = obj.ids

    if pooled:
        # we need to repeat each element in the counts and IDs by the number
        # of time points picked for each path
        counts = np.repeat(counts, pooled)
        ids = np.repeat(ids, pooled)

    # Find the objects seed number and statistical replicate number
    seed_number = obj.filename.split("mldata/seed")[-1].split("/")[0]
    replicate_number = obj.filename.split("mldata/seed")[-1].split("reactive")[-1].split("_r")[-1].split("/")[0]
    seeds = np.array([seed_number]*counts.shape[0])
    replicates = np.array([replicate_number]*counts.shape[0])

    # Add the counts, IDs, seeds, and replicates to the averaged data
    counts = counts.reshape((counts.shape[0], 1))
    ids = ids.reshape((ids.shape[0], 1))
    seeds = seeds.reshape((seeds.shape[0], 1))
    replicates = replicates.reshape((replicates.shape[0], 1))

    if pooled:
        # then we must also add the time point labels to the data
        time_labels = obj.sampled_time_labels
        time_labels = time_labels.reshape((time_labels.shape[0], 1))
        data = np.concatenate([data, counts, ids, seeds, replicates, time_labels], 1)
    else:
        data = np.concatenate([data, counts, ids, seeds, replicates], 1)

    # Return the average
    return data

"""
Adds the trajectory IDs as a feature to the averaged data
"""
def add_ids_as_feature(obj):
    """
    Inputs:
    obj - A loaded and time averaged Features object
    Outputs:
    data - the time-average data with trajectory IDs appended (matrix with trajectories as rows and
    features as columns)
    """

    # Get the averaged data
    data = obj.averaged_data
    # Get the counts for each trajectory
    ids = obj.ids
    # Add the counts to the averaged data
    ids = ids.reshape((ids.shape[0], 1))
    data = np.concatenate([data, ids], 1)
    # Return the average
    return data

"""
Puts trajectory data from different seeds but with the same order parameter
together into the same matrix. Also adds a column for an id
unique to the seed of that trajectory.
"""
def concatenate_trajectories(datas):
    """
    Inputs:
    datas - a list of matrices of time averaged trajectory data (with trajectories  or frames as rows and
    features as columns)
    Outputs:
    data - a matrix joining the data from all trajectories of the same order (again with trajectories or frames as
    rows and features as columns)
    """

    # Return the concatenated data matrices
    data = np.concatenate(datas, 0)
    return data

"""
Creates a pandas data frame with all trajectories of a given order, with data
averaged over a time window
"""
def create_df(order, seeds='*', replicates='*', time=None, pooled=False, loc="./", num_shots=None, subsample=None, file_prefix=""):
    """
    Inputs:
    order - the order parameter of the objects to load (float)
    time - A range of time points between which to average the features for each trajectory (list)
           For example, time = [-200,100] will grab the 200 time points
           preceding t = 0 through the 100 time points following t = 0
           Default (None) uses all available time points
    seeds - the numbers of the seeds of the objects to load (defaults to loading all seeds'). String, int, or list of strings and/or integers
    replicates - the numbers of the replicates of the objects to load (defaults to loading all replicates). String, int, or list of strings and/or integers
    pooled - if false, then each pathway's features are averaged between the time range. Else, if a number n,
             then n random time points' feautres are sampled between the time range.
    loc - the path to the folder containing the ./tis_mldata/ folder containing
          the data on which to run this script's analysis
    num_shots - the number of shots to include from each ensemble. Int or None. None, the default,
                will use all available shots
    series - ignored when pooled is False. Else, this indicates that the time window should be loaded in as a series or sequence 
             of time points (not randomly sampled or shuffled) when True. When False (default), time points will be randomly sampled
             and ordered. Should set series == True when compiling data for sequential models such as RNNs. When series is True, pooled 
             can be any positive number, and the entire time window will still be grabbed

    Outputs:
    df - a data frame containing all trajectories of the given order, averaged
    over the specified time window (pandas data frame with trajectories as rows and
    features as columns)
    """

    # Load the objects of the given order
    objs = load_objs(order, seeds=seeds, replicates=replicates, loc=loc, num_shots=num_shots, subsample=subsample, file_prefix=file_prefix)

    if not pooled:
        # Average over the given time points for each feature in each trajectory
        objs = [average_over_time_points(obj, time=time) for obj in objs]
    else:
        # Randomly sample pooled number of time points from each path
        objs = [sample_between_time_points(obj, pooled=pooled, time=time) for obj in objs]


    # Add the trajectory counts and pathway IDs to the feature matrix in each object
    datas = [add_counts_and_ids_and_seeds_and_replicates_as_features(obj, pooled=pooled) for obj in objs]

    # Put all the data together into a single matrix
    datas = concatenate_trajectories(datas)

    # Turn the matrix into a pandas data frame data, counts, ids, seeds, replicates
    if pooled:
        df = pd.DataFrame(datas, columns = objs[0].ft_names.tolist() + ['Counts', 'IDs', 'Seed', 'Replicate', 'Time point'])
    else:
        df = pd.DataFrame(datas, columns = objs[0].ft_names.tolist() + ['Counts', 'IDs', 'Seed', 'Replicate'])

    # Add the order as a column with identical elements, in preparation for combining
    # this matrix with matrices from other trajectories
    df['Order'] = order

    # convert all values to float, if not already
    df = df.astype(np.float64)

    return df

"""
Load in and set up the different variants' transition pathway data formatted for 
logistic regression, linear regression, or fully connected neural net
"""
def set_up_data(variants_summary, variants, order_parameters, time, seeds, replicates, pooled, num_shots, subsample, loc='./'):
    variant_dfs = []
    for variant in variants:
        # pull out variant-specific data
        temp_df = variants_summary.loc[variants_summary['Variant'] == variant]
        iteration = temp_df['Iteration'].to_list()[0]
        kcat_avg = float(temp_df['TIS raw k'].to_list()[0].split('(')[0])
        kcat_sem = float(temp_df['TIS raw k'].to_list()[0].split('(')[-1].split(')')[0])

        # determine where variant's data are located
        if variant == 'WT':
            loc = '/data/karvelis02/tis/'
        else:
            loc = f'/data/karvelis02/comets_designs/rates/design_iteration_{iteration}/{variant.lower()}/'
    
        # check if data is available for variant; if not, skip
        if not os.path.exists(loc+'tis_mldata/seed1/reactive_r1/output/features.obj'):
            continue
        
        # load data for variant, one order at a time
        dfs = []
        for order in order_parameters:
            df = create_df(order, time=time, seeds=seeds, replicates=replicates,
                           pooled=pooled, loc=loc, num_shots=num_shots, subsample=subsample)
            
            # add a columns for the variant's mutation, design iteration, and activity
            df['Iteration'] = [iteration]*df.shape[0]
            df['Variant'] = [variant]*df.shape[0]
            df['kcat AVG'] = [kcat_avg]*df.shape[0]
            df['kcat SEM'] = [kcat_sem]*df.shape[0]

            dfs.append(df)

        # join variant's data into single dataframe 
        variant_df = pd.concat(dfs, axis=0)
        variant_dfs.append(variant_df)

    df = pd.concat(variant_dfs, axis=0)
    return df

"""
Load in and set up the different variants' transition pathway data formatted for 
recurrent models such as LSTMs or other RNNs. This class is useful for storing 
and organizing such data
"""
class Recurrent_data():
    # Purpose: class for an object that organizes recurrent (time series) data for 
    #          use in recurrent, sequence-based model pipelines. Because recurrent 
    #          data is 3 dimensional: [pathways, time steps, features], it can't be 
    #          easily stored in pandas dataframes, so this class is used instead to 
    #          define an object.
    # Inputs:
    # shape -- shape of the data matrix, X, which will be of form
    #          [pathways, time points, features], and pathways can be
    #          set to 0 for initialiation of empty array. E.g.,
    #          shape = (0,30,70) for a time window of 30 fs with 70 features 
    def __init__(self, shape, ft_names):
        self.data = np.ndarray(shape)
        self.variant = []
        self.kcat = []
        self.kcat_sem = []
        self.order = []
        self.seed = []
        self.replicate = []
        self.ids = []
        self.ft_names = ft_names
        self.timesteps = []
    
    def info(self):
        print ("self.data -- data matrix, X of form [pathways, time points, features]")
        print ("self.variant -- list of variant identities for each entry along axis 0 of self.data")
        print ("self.kcat -- list of variant's TIS-based kcat AVG for each entry along axis 0 of self.data")
        print ("self.kcat_sem -- list of variant's TIS-based kcat SEM for each entry along axis 0 of self.data")
        print ("self.order -- list of pathway type's order parameter entry along axis 0 of self.data")
        print ("self.seed -- list of pathway's seed paths along axis 0 of self.data")
        print ("self.replicate -- list of pathway's replicate number along axis 0 of self.data")
        print ("self.ids -- list of the pathway's IDs along axis 0 of self.data")
        print ("self.timesteps -- array of the time steps labels along axis 1 of self.data")

"""
Compile data across different variants into a format that can be used by sequential, recurrent models
"""
def set_up_data_recurrent(variants_summary, variants, load_file=False, save_filename='data.data', order_parameters=[0.8],
                          seeds='*', replicates='*', subsample=100, num_shots=None, pooled=1):
    """
    Inputs:
    order - the order parameter of the objects to load (float)
    time - A range of time points between which to average the features for each trajectory (list)
           For example, time = [-200,100] will grab the 200 time points
           preceding t = 0 through the 100 time points following t = 0
           Default (None) uses all available time points
    seeds - the numbers of the seeds of the objects to load (defaults to loading all seeds'). String, int, or list of strings and/or integers
    replicates - the numbers of the replicates of the objects to load (defaults to loading all replicates). String, int, or list of strings and/or integers
    pooled - if false, then each pathway's features are averaged between the time range. Else, if a number n,
             then n random time points' feautres are sampled between the time range.
    loc - the path to the folder containing the ./tis_mldata/ folder containing
          the data on which to run this script's analysis
    num_shots - the number of shots to include from each ensemble. Int or None. None, the default,
                will use all available shots
    series - ignored when pooled is False. Else, this indicates that the time window should be loaded in as a series or sequence 
             of time points (not randomly sampled or shuffled) when True. When False (default), time points will be randomly sampled
             and ordered. Should set series == True when compiling data for sequential models such as RNNs. When series is True, pooled 
             can be any positive number, and the entire time window will still be grabbed
    load_file - when False (default), data will be freshly compiled and saved to save_filename.
                when not False, must be a string of the filename from which to load old data
                that was already compiled and saved somewhere. This argument provides the option
                to load pre-compiled data that was previously saved from an earlier run.
    save_filename - when load_file is False, the freshly compiled Data object will be saved 
                    to save_filename. Can load in later using:
                    with open(save_filename, 'rb') as f:
                        Data = pickle.load(f)
    Outputs:
    df - a data frame containing all trajectories of the given order, averaged
    over the specified time window (pandas data frame with trajectories as rows and
    features as columns)
    """
    if load_file and os.path.exists(load_file):
        with open(load_file, 'rb') as f:
            Data = pickle.load(f)
        return Data
        
    Data = 0
    variant_dfs = []
    for variant in variants:
        # pull out variant-specific data
        temp_df = variants_summary.loc[variants_summary['Variant'] == variant]
        iteration = temp_df['Iteration'].to_list()[0]
        kcat_avg = float(temp_df['TIS raw k'].to_list()[0].split('(')[0])
        kcat_sem = float(temp_df['TIS raw k'].to_list()[0].split('(')[-1].split(')')[0])

        # determine where variant's data are located
        if variant == 'WT':
            loc = '/data/karvelis02/tis/'
        else:
            loc = f'/data/karvelis02/comets_designs/rates/design_iteration_{iteration}/{variant.lower()}/'
    
        # check if data is available for variant; if not, skip
        if not os.path.exists(loc+'tis_mldata/seed1/reactive_r1/output/features.obj'):
            continue


        # Load the objects of the given order
        for order in order_parameters:
            objs = load_objs(order, seeds=seeds, replicates=replicates, loc=loc,
                             num_shots=num_shots, subsample=subsample)

            # Splice out the desired time points for each path
            objs = [sample_between_time_points(obj, pooled=pooled, time=time, series=True) for obj in objs]
            
            # Append objs data to Data
            if Data == 0:
                Data = Recurrent_data((0,objs[0].averaged_data.shape[1],objs[0].averaged_data.shape[2]), objs[0].ft_names)
                Data.timesteps = np.arange(time[0], (time[1]+1))
            for obj in objs:
                seed_number = int(obj.filename.split('/seed')[-1].split('/')[0])
                rep_number = int(obj.filename.split('/output/features')[0].split('_r')[-1])

                Data.seed.extend([seed_number]*obj.averaged_data.shape[0])
                Data.replicate.extend([rep_number]*obj.averaged_data.shape[0])
                Data.ids.extend(obj.ids)
                Data.variant.extend([variant]*obj.averaged_data.shape[0])
                Data.order.extend([order]*obj.averaged_data.shape[0])
                Data.kcat.extend([kcat_avg]*obj.averaged_data.shape[0])
                Data.kcat_sem.extend([kcat_sem]*obj.averaged_data.shape[0])
                Data.data = np.append(Data.data, obj.averaged_data, axis = 0)
    
    # save data to file
    # pickle.dump(Data, open(save_filename, 'wb'), protocol=4) ###

    ### Look into saving the data as a memory-mapped numpy array, so 
    ### that you can later stream it piece-by-piece during analysis
    ### More info: https://discuss.pytorch.org/t/memory-efficient-data-streaming-for-larger-than-memory-numpy-arrays/11928
    ### HDF5 files are another option:
    ### More info: https://stackoverflow.com/questions/30329726/fastest-save-and-load-options-for-a-numpy-array

    """
    save as a memory mapped array, we'll add the metadata like variant, order, 
    kcat, kcat err as the last four columns
    """
    # initialize the memory mapped array
    ax0 = Data.data.shape[0]
    ax1 = Data.data.shape[1]
    ax2 = Data.data.shape[2]
    fp = np.memmap(save_filename.replace('.data',f'.{ax0}-{ax1}-{ax2}memnpy'), dtype='float32', mode='w+', shape=(ax0,ax1,ax2))
    # NOTE: load using something like >> fp = np.memmap('filename.20000-111-72memnpy', dtype='float32', mode='r', shape=(20000,111,72))

    # fill in the data matrix
    fp[:,:,:] = Data.data

    fout.write(f'\nData shape [pathways, timepoints, features]: {Data.data.shape}\n\n')
    fout.flush()

    # flush the changes to the disk
    fp.flush()

    # save only the metadata 
    Data.data = None
    pickle.dump(Data, open(save_filename.replace('.data','.metadata'), 'wb'), protocol=4)


    
    return Data

"""
Helper function to generate descriptive name for data file
"""
def gen_data_filename():
    
    order_label = 'o'
    for order in order_parameters:
        order_label += str(order).replace('.','dot') + '_'

    seed_label = 's'
    if seeds == '*':
        seed_label += 'all_'
    else:
        for s in seeds:
            seed_label += str(s) + '_'

    rep_label = 'r'
    if replicates == '*':
        rep_label += 'all_'
    else:
        for r in replicates:
            rep_label += str(r) + '_'

    time_label = 't'
    for t in time:
        time_label += str(t) + '_'

    sub_label = 'sub' + str(subsample) + '_'
    num_label = 'num' + str(num_shots)

    if not os.path.exists('./data/'):
        os.mkdir('./data/')
    filename = f'./data/tp{str(tp_data).lower()}_gs{str(gs_data).lower()}_{order_label}{seed_label}{rep_label}{time_label}{sub_label}{num_label}.data'

    return filename


if __name__ == "__main__":

    """
    SETTINGS
    """
    if True:
        # Defaults:
        tp_data, gs_data = True, True
        load_file = False
        save_filename = 'data.data'
        variants_summary = './input/variants_summary.csv'
        random_seed = 333
        num_shots, subsample = None, None
        replicates, seeds = '*','*'
        load_file, save_filename = None, None
        output_text = './prep_data_output.txt'
        package_loc = './'
        with open(sys.argv[1], 'r') as f:
            settings = f.read()
            exec(settings)

            np.random.seed(random_seed)

            if load_file == None:
                load_file = gen_data_filename()
            if save_filename == None:
                save_filename = gen_data_filename()

        sys.path.append(package_loc + 'input/')
        import feature_extraction_submit

        variants_summary = pd.read_csv(variants_summary)
        if variants == '*':
            variants_summary = variants_summary.loc[variants_summary['TIS raw k'] != 'nan (nan)']
            variants = variants_summary['Variant'].to_list()


    # Create an output file to log progress
    fout = open(output_text, 'w')
    fout.write(f'Python: {sys.version}\n')
    fout.write(f'Total variants: {len(variants)}, but will only use those with all required data\n\nLoading data...\n')
    fout.flush()

    """
    Load data -- can specify to load from file, as opposed to compiling anew each time; see
    function description 
    """
    Data = set_up_data_recurrent(variants_summary, variants, order_parameters=order_parameters, seeds=seeds,
                                 replicates=replicates, subsample=subsample, num_shots=num_shots,
                                 load_file=load_file, save_filename=save_filename)
    fout.write('Finished loading data\n')
    fout.write(f'{np.unique(np.array(Data.variant)).shape[0]} variants will be used\n')
    included_variants = np.unique(np.array(Data.variant))
    fout.write(f'Variants:\n{included_variants}\n\n')
    fout.flush()


    # if classification:
    #     None
    #     # # make Y discrete:
    #     # wt_kcat = Data.kcat[np.argwhere(np.array(Data.variant)=='WT')[0][0]]
    #     # Y = np.array([1 if ((y - wt_kcat) > 1) else 0 for y in Data.kcat])
    #     # fout.write(f'Class 0 count: {np.sum(Y==0)}\n')
    #     # fout.write(f'Class 1 count: {np.sum(Y==1)}\n')
    # fout.write('\nRunning cross validation...\n')
    # fout.flush()

    # """
    # Execute training and testing for cross validation of model 
    # """
    # histories, others = kcat_pred_recurrent(Data, noisy_kcat=noisy_kcat, k=k, seed=seed, classification=classification, 
    #                                         features=features, split_by_variant=split_by_variant, epochs=epochs, dropout=dropout,
    #                                         recurrent_dropout=recurrent_dropout, lstm_units=lstm_units, normalize=normalize)

    # fout.write('\nCross validation finished\nWriting scores:\n')
    # # write output to text file
    # metrics = ['accuracy', 'auc']
    # for m in metrics:
    #     fout.write(f'{m}: {histories[m][-1]}\n')
    #     fout.write(f"val_{m}: {histories['val_'+m][-1]}\n")
    #     fout.flush()

    # if split_by_variant:
    #     fout.write(f'others: {others}\n')

    # fout.write(f'\nSaving complete scores report to\n{score_file}\n')
    # fout.flush()
    # with open(score_file, 'wb') as f:
    #     pickle.dump(histories, f)
    #     # can load using:
    #     # with open(score_file, 'rb') as f:
    #     #   histories = pickle.load(f)



    # fout.write("\n\nDONE. SUCCESSFUL\n")
    # fout.flush()
    # fout.close()
