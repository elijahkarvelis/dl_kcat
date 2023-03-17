#############################################
# AUTHOR: Elijah Karvelis (karvels2@mit.edu)
# 9/24/20
#############################################
"""
NOTE: this script is the same as /data/karvelis/tis_v7/tis_mldata/feature_extraction_submit_deprecated.py,
except that this one reads in a config txt file for its settings
This script performs "feature" extraction on raw DCD files from CHARMM.
It calls scripts to read in DCD's within some directory, calculate features
at each time step, and ultimately save a large numpy array of form
[pathways, timesteps, features], as well as two other numpy arrays that store
the order of the pathways' id's and also the order of the features (more info
below). This script calls functions that DO time alignment of trajectories.

Will submit batch_size number of jobs simultaneously, wait for them all to complete,
and will then submit the next batch of files.

You can run this script multiple times on the same directory using the exact same
execution at the command line (i.e., the same arguments), it won't redo any
analyses or accidentally delete previously processed data. This is a useful feature
for when you wish to get a head-start on DCD post-processing before even finishing
the sampling of the ensemble of DCDs. For example, if you're generating and writing
data to directory X over the course of several weeks, you can run this script on
directory X multiple times (say, at days 1, 2, 4, 55, etc...) to process
whatever extra trajectories had been generated since the last time you ran this script.
So, you can process your data "live," as it's being generated.

Run like so:
python feature_exraction_submit.py feature_extraction_submit_config.txt

where feature_extraction_config.txt specifies each input (see below). Each
input is on its own line, followed by " = " and its value

It is suggested that you submit this script to Thor using the Bash (.sh) script:
/data/karvelis/tis_package/submit_feature_extraction_submit.sh
when you submit the above bash script, which calls this python script, use the submit script:
/data/karvelis/tis_package/input/scripts/submit.maui.infiniteh_32p.exmem
which will dedicate 2 nodes with 16 processors each to the job, for inifinite time

INPUTS
batch_size:  the number of DCD files to submit to Thor for post-processing at a time
             25 is recommended, I wouldn't go any higher
rev_l:       how many timesteps before t=0 to include
for_l:       how many timesteps after t=0 to include
input_dir:   the name of the directory with the DCD files you want to process.
             It's ok if the .vel files are in the same location
output_file: the name of the file you want for all the output data (the array of form
             [pathways, timesteps, features])
output_text: name of the text file to which you want output/progress written
crd_filename:the .CRD file of the basic, starting structure for your protein. If you
             trimmed your ML data, then you need to enter the name of a trimmed CRD file
             here, otherwise use a standard CRD file
inc:         DON'T SPECIFY (leave blank instead). Meant to specify the frequency
             of calculating/saving features, but this code has not been completed.
             Instead, features are calcuated at EVERY timepoint

Example feature_extraction_submit_config.txt file
batch_size = 25
rev_l = 300
for_l = 100
input_dir = /data/karvelis/tis_v7/dylan/seed3/reactive_r1/output/lam_1_traj/
output_file = /data/karvelis/tis_v7/dylan/seed3/reactive_r1/output/features.npy
output_text = /data/karvelis/tis_v7/dylan/seed3/reactive_r1/output/feature_extraction_progress_text.txt
crd_filename = /data/karvelis/tis_v7/trimtest.crd

##### OUTPUT #####
# Some new directory within the working directory that has four files:
# 1: a saved numpy array of the features [pathways, timepoints, features]: has name output_file
# 2: a saved numpy array that is a list of the pathway ids in the same order as
#    they are in array 1 [pathways]: pathway_ids.npy
# 3: a saved numpy array that is a list of the features' names in the same order as
#    they are in array 1 [features]: feature_names.npy
# 4: a saved python object in the same directory as output_file but with the name features.obj,
#    it can be loaded by opening the file with option 'rb', and using pickle.load on it:
#         with open('/path/to/features.obj', 'rb') as f:
#              features = pickle.load(f)
# NOTE: the script that loads the object must have first iport the Features class, like so:
# sys.path.append("/data/karvelis/tis_package_fbeta/feature_extraction/")
# from feature_extraction_submit import Features
"""

# import modules for submitting to Thor
import os
import subprocess
from glob import glob
from math import ceil
import time
import numpy as np
import sys
import pickle
sys.path.append("/data/karvelis/tis_package/feature_extraction/")
from mldata_stats import *
from cut_reactives import cut_reactives

# define functions
def save_feature_names(output_file):
    feature_names = []
    feature_names.append('Dist AC6/O2,NDP/N7N')
    feature_names.append('Dist AC6/O2,NDP/O7N')
    feature_names.append('Dist AC6/O3,MG6/H24')
    feature_names.append('Dist AC6/O6,MG6/M16')
    feature_names.append('Dist AC6/O8,GLU496/HE2')
    feature_names.append('Dist AC6/O8,MG6/M17')
    feature_names.append('Dist GLU319/OE1,AC6/C5')
    feature_names.append('Dist MG6/H25,AC6/O6')
    feature_names.append('Dist MG6/H26,AC6/O6')
    feature_names.append('Dist MG6/H27,AC6/O6')
    feature_names.append('Dist MG6/H28,AC6/O6')
    feature_names.append('Dist MG6/H31,AC6/O6')
    feature_names.append('Dist MG6/H32,AC6/O6')
    feature_names.append('Dist MG6/M16,AC6/O3')
    feature_names.append('Dist MG6/M17,AC6/O6')
    feature_names.append('Dist NDP/H4N2,AC6/C4')
    feature_names.append('Ang AC6/O6,MG6/M16,AC6/O3')
    feature_names.append('Ang AC6/O8,MG6/M17,AC6/O6')
    feature_names.append('Ang MG6/M17,AC6/O6,MG6/M16')
    feature_names.append('Dist AC6/C1,AC6/C4')
    feature_names.append('Dist AC6/C1,AC6/O2')
    feature_names.append('Dist AC6/C1,AC6/O3')
    feature_names.append('Dist AC6/C4,AC6/C7')
    feature_names.append('Dist AC6/C4,AC6/O6')
    feature_names.append('Dist AC6/C5,AC6/C4')
    feature_names.append('Dist AC6/C5,AC6/C7')
    feature_names.append('Dist AC6/C7,AC6/C9')
    feature_names.append('Dist AC6/C7,AC6/O8')
    feature_names.append('Ang AC6/C1,AC6/C4,AC6/C7')
    feature_names.append('Ang AC6/C4,AC6/C7,AC6/C5')
    feature_names.append('Ang AC6/C4,AC6/C7,AC6/C9')
    feature_names.append('Ang AC6/C5,AC6/C4,AC6/C1')
    feature_names.append('Ang AC6/C5,AC6/C7,AC6/C9')
    feature_names.append('Dihe AC6/C1,AC6/C5,AC6/C7,AC6/C4')
    feature_names.append('Dihe AC6/C5,AC6/C4,AC6/C7,AC6/C9')
    feature_names.append('Dist NDP/H4N2,NDP/C4N')
    feature_names.append('Dist NDP/N7N,NDP/O2N')
    feature_names.append('Ang NDP/C4N,NDP/N1N,NDP/C1NQ')
    feature_names.append('Ang NDP/C6N,NDP/C3N,NDP/C7N')
    feature_names.append('Ang NDP/N7N,NDP/H72N,NDP/O2N')
    feature_names.append('Dihe NDP/C2N,NDP/C3N,NDP/C7N,NDP/N7N')
    feature_names.append('Dihe NDP/C2NQ,NDP/C1Nq,NDP/N1N,NDP/C6N')
    feature_names.append('Dihe NDP/C4N,NDP/C3N,NDP/C7N,NDP/O7N')
    feature_names.append('Dihe NDP/H1NQ,NDP/C1NQ,NDP/N1N,NDP/C2N')
    feature_names.append('Dist MG6/O18,MG6/M17')
    feature_names.append('Dist MG6/O19,MG6/M17')
    feature_names.append('Dist MG6/O20,MG6/M17')
    feature_names.append('Dist MG6/O21,MG6/M16')
    feature_names.append('Dist MG6/O22,MG6/M16')
    feature_names.append('Ang MG6/H23,MG6/O22,MG6/M16')
    feature_names.append('Ang MG6/H24,MG6/O22,MG6/M16')
    feature_names.append('Ang MG6/H25,MG6/O21,MG6/M16')
    feature_names.append('Ang MG6/H26,MG6/O21,MG6/M16')
    feature_names.append('Ang MG6/H27,MG6/O20,MG6/M17')
    feature_names.append('Ang MG6/H28,MG6/O20,MG6/M17')
    feature_names.append('Ang MG6/H29,MG6/O18,MG6/M17')
    feature_names.append('Ang MG6/H30,MG6/O18,MG6/M17')
    feature_names.append('Ang MG6/H31,MG6/O19,MG6/M17')
    feature_names.append('Ang MG6/H32,MG6/O19,MG6/M17')
    feature_names.append('Dist GLU496/OE2,GLU496/HE2')
    feature_names.append('Dist GLN136/NE2,NDP/O7N')
    feature_names.append('Dist MG6/H25,MG6/O21')
    feature_names.append('Dist MG6/H26,MG6/O21')
    feature_names.append('Dist MG6/H27,MG6/O20')
    feature_names.append('Dist MG6/H28,MG6/O20')
    feature_names.append('Dist MG6/H31,MG6/O19')
    feature_names.append('Dist MG6/H32,MG6/O19')
    feature_names.append('Ang GLN136/NE2,GLN136/HE22,NDP/O7N')
    feature_names.append('Dihe AC6/O6,AC6/C4,AC6/C5,AC6/C5-H')
    feature_names.append('Dihe AC6/O8,AC6/C7,AC6/C5,AC6/C5-H')

    # convert list to numpy array:
    feature_names = np.array(feature_names)

    # save numpy array:
    save_dir = '/'.join(output_file.split('/')[0:-1]) + '/'
    np.save(save_dir+'feature_names.npy', feature_names)

    return feature_names, save_dir+'feature_names.npy'

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

def save_obj(output_file, pathway_ids_filename, rev_l):

    fts_obj_filename = ('/').join(output_file.split('/')[0:-1]) + '/features.obj'
    ft_object = Features(output_file, pathway_ids_filename, rev_l)

    with open(fts_obj_filename, 'wb') as output:
        pickle.dump(ft_object, output) # load using pickle.load(fts_obj_filename)

    return fts_obj_filename

if __name__ == "__main__":
    # python script that pulls out features and saves as npy array. Current file
    # does time alignemnt and finds all of the exact same features that Brian Bonk
    # considered for KARI.
    extract_features_file = '/data/karvelis/tis_package/feature_extraction/feature_extraction.py'

    # This CRD file is specific for KARI
    #crd_filename = '/bt/home/karvelis/Documents/Research/KARI/Structure_Prep/minimization/output/minimized_v4.crd'

    # Grab settings from configuration file
    config_file = sys.argv[1]
    settings_dict = {}
    with open(config_file, 'r') as f:
        for line in f:
            line = line.rsplit()
            settings_dict[line[0]] = line[-1]

    batch_size = int(settings_dict['batch_size'])
    rev_l = int(settings_dict['rev_l'])
    for_l = int(settings_dict['for_l'])
    input_dir = settings_dict['input_dir']
    output_file = settings_dict['output_file']
    output_text = settings_dict['output_text']
    crd_filename = settings_dict['crd_filename']
    try:
        inc = int(settings_dict['inc'])
    except:
        inc = 1


    # will
    #submit_path = '/data/karvelis/tis_package/input/scripts/submit.maui.infiniteh_32p.exmem'
    #if not (os.path.exists(input_dir+'submit/')):
    #    os.mkdir(input_dir+'submit/')
    #bash_submit_path = input_dir+'submit/'

    output_text = open(output_text, 'w')



    save_feature_names(output_file)


    # We're only going to process the DCD files that haven't already been
    # processed -- this allows you to run this post-processing feature extraction
    # script on the same directory, multiple times, while data is still being
    # generated and written to that directory!
    all_files = glob(input_dir + 'traj_*-pad_.dcd')
    ###npy_files = glob(input_dir + 'traj_*-pad_.npy')

    ###processed_ids = []
    ###for npy_file in npy_files:
    ###    idd = npy_file.split('/')[-1].split('-')[0].split('_')[-1]
    ###    processed_ids.append(idd)

    pathway_ids_filename = ('/').join(output_file.split('/')[0:-1]) + '/pathway_ids.npy'
    if os.path.exists(pathway_ids_filename):
        final_pathway_ids = np.load(pathway_ids_filename)
    else:
        final_pathway_ids = []

    files = []
    for dcd_file in all_files:
        idd = dcd_file.split('/')[-1].split('-')[0].split('_')[-1]

        if int(idd) == 0: # only count ID 0 if it was accepted -- Output class checks this
            save_dir = '/'.join(output_file.split('/')[0:-1]) + '/'
            output_text_filename = save_dir + 'output_text.txt'
            output = Output(output_text_filename)
            output.set_up()
            if not (0 in output.ids):
                # if 0 is not in the Output class' list of IDs, then it was not accepted
                # and we should not include it for feature extraction
                continue

        if not (idd in final_pathway_ids):
            files.append(dcd_file)

    total_files = len(files)
    num_epochs = ceil(total_files/batch_size)

    file_count = 0

    for i in range(num_epochs):

        try:
            file_batch = files[(i*batch_size) : (i*batch_size + batch_size)]
        except IndexError:
            file_batch = files[(i*batch_size)::]

        processes = []
        for filename in file_batch[0:-1]:
            filename = filename.split('/')[-1]
            out_name = filename.split('.')[0] + '.npy'


            ### new ##
            #command = ["python", extract_features_file] +
            #          [str(v) for v in (input_dir+filename),
            #          crd_filename, rev_l, for_l, inc] +
            #          ["ax=None"]
            #exec = subprocess.Popen(command, stdout=subprocess.PIPE)
            #########


            # start a new process to calculate features for the DCD file:
            exec = subprocess.Popen("python " + extract_features_file + ' ' + input_dir+filename + ' ' +
                                    crd_filename + ' ' + str(rev_l) + ' ' + str(for_l) + ' ' + str(inc), shell=True)

            # append the execution to the list of running processes:
            processes.append(exec)
            file_count += 1

        # run the last file in the batch
        filename = file_batch[-1].split('/')[-1]
        out_name = filename.split('.')[0] + '.npy'


        ### new ##
        #command = ["python", extract_features_file] +
        #          [str(v) for v in (input_dir+filename),
        #          ["ax=None"]
        #exec = subprocess.Popen(command, stdout=subprocess.PIPE)
        #########


        # start a new process to calculate features for the DCD file:
        exec = subprocess.Popen("python " + extract_features_file + ' ' + input_dir+filename + ' ' +
                                crd_filename + ' ' + str(rev_l) + ' ' + str(for_l) + ' ' + str(inc), shell=True)

        # append the execution to the list of running processes:
        processes.append(exec)
        file_count += 1

        # force the program to wait for all processes to finish before proceeding:
        for i in range(len(processes)):
            exec = processes[i]
            exec.wait(timeout=None)


        # Now stitch all the npy files from the batch into a single npy file
        pathway_ids = []
        flag = 0
        for filename in file_batch:
            filename = filename.split('/')[-1]
            filename = filename.split('.')[0] + '.npy'

            feature_array = np.load(input_dir + filename)
            feature_array = np.reshape(feature_array, (1, feature_array.shape[0], feature_array.shape[1]))

            # add the array to the stitched array with all pathways
            if flag == 0:
                flag = 1
                stitch_array = feature_array
            else:
                #print (filename)
                #print (stitch_array.shape)
                #print (feature_array.shape)
                stitch_array = np.vstack((stitch_array,feature_array))

            # add the pathways ids (in correct order) to the pathway_ids list
            id = filename.split('-')[0].split('_')[-1]
            pathway_ids.append(id)


        pathway_ids = np.array(pathway_ids)
    ###    pathway_ids_filename = ('/').join(output_file.split('/')[0:-1]) + '/pathway_ids.npy'

        if os.path.exists(output_file):
            # then you have run this script for this input_dir before or you're past the
            # first batch, and there is an existing output_file (stitched .npy array)
            # to which we should append our newly processed data at the end.

            final_npy = np.load(output_file)

            final_npy = np.vstack((final_npy,stitch_array))

            np.save(output_file, final_npy)

            final_pathway_ids = np.load(pathway_ids_filename)
            final_pathway_ids = np.concatenate((final_pathway_ids, pathway_ids))
            np.save(pathway_ids_filename, final_pathway_ids)

        else:
            np.save(output_file, stitch_array)
            np.save(pathway_ids_filename, pathway_ids)

        output_text.write("Have processed %s of %s files so far...\n" %(file_count,total_files))
        output_text.flush()

    # save a Python object that summarizes the data, the ids, the counts
    fts_obj_filename = save_obj(output_file, pathway_ids_filename, rev_l)

    # remove reactive pathways from features.obj if it's a non-reactive esemble:
    if "almost_reactive" in output_file:
        cut_reactives(fts_obj_filename)

    output_text.write("Features object saved to \n%s\n" %(fts_obj_filename))
    output_text.flush()
    output_text.close()

    # run general stats and make figures for ensemble
    inspect(fts_obj_filename, trange=[-50,0], lam_trange=[-200,100], e_trange=[-200,200])
