"""
E Karvelis
9/13/2023

Submits a set of jobs to complete a grid search-like hyperparameter
optimization.

"""




# import modules:
import os
import shutil
import time as time_sleep
import subprocess
from datetime import datetime

# define functions
def replace_line(file_name, line_num, text):
    with open(file_name, 'r') as fff:
        lines = fff.readlines()
    #lines = open(file_name, 'r').readlines()
    lines[line_num] = text
    out = open(file_name, 'w')
    out.writelines(lines)
    out.close()

def check_job(job_id):
    # pass this a string of the job ID, and this will return
    # true/false (bool) whether the job is still running
    proc = subprocess.Popen(["qstat"], stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()
    out = str(out)

    if job_id in out:
        running = True
    else:
        running = False

    return running

def check_jobs():
    # reads job_wrapper_job_ids.txt and counts how many of its
    # jobs are still running

    running_jobs = 0
    with open("job_wrapper_job_ids.txt", 'r') as f:
        for line in f:
            if ".thor" in line:
                job_id = str(line.rsplit()[0])
                if check_job(job_id):
                    running_jobs += 1

    return running_jobs

def write_config_and_submit_files(model='transformer',
                                  task='kcat regression',
                                  split_by_variant='True',
                                  regularization='None',
                                  mixed_variants='False',
                                  model_type='test'):
    # writes unique {model}_1_config.txt and submit_{model}_1_tmp.sh
    # for execution of job with specified parameters.
    def reg_label(reg):
        if str(reg) == 'None':
            return 'None'
        else:
            return f"""{reg.split('{')[-1].split(':')[0].replace(' ','').replace("'","")}-{reg.split(':')[-1].split('}')[0].replace(' ','')}"""

    task_labels = {'kcat regression': 'reg',
                   'NR/R binary classification': 'nr-r',
                   'S/F binary classification': 's-f'}

    subfolder = f"{os.getcwd()}/m{model}_t{task_labels[task]}_sbv{split_by_variant}"+\
                f"_r{reg_label(regularization)}_mv{mixed_variants}_{model_type}/"
    if not os.path.exists(subfolder):
        os.mkdir(subfolder)
    config_filename = f'{subfolder}{model}_1_config.txt'
    submit_filename = f'{subfolder}submit_{model}_1_tmp.sh'

    # write the config file, which specifies job parameters
    with open(config_filename, 'w') as f:
        f.write(f"data_file = '/data/karvelis03/dl_kcat/data/total/tptrue_gsfalse_o-0dot4_0dot8_s1_2_3_4_5_r1_2_t-160_-130_sub500_numNone.550000-31-70memnpy'\n")
        f.write(f"meta_file = '/data/karvelis03/dl_kcat/data/total/tptrue_gsfalse_o-0dot4_0dot8_s1_2_3_4_5_r1_2_t-160_-130_sub500_numNone.550000-31-70metadata'\n")
        f.write("loc = '/data/karvelis03/dl_kcat/'\n")
        f.write("path_set_size = 10\n")
        f.write(f"batch_size = 32\n")
        f.write("cv_folds = 3\n")
        f.write("epochs = 10\n\n")

        if model == 'transformer':
            f.write(f"d_model = 10\n")
            f.write(f"d_input_enc = 10\n")
            f.write(f"n_head = 2\n")
            f.write(f"d_tran_ffn = 20\n")
            f.write(f"dropout_tran_encoder = 0.2\n")
            f.write(f"n_tran_layers = 1\n")
            f.write(f"d_mlp_head = 10\n")
            f.write(f"dropout_mlp_head = 0.2\n\n")
        elif model == 'lstm':
            f.write(f"lstm_hidden_size = 10\n")
            f.write(f"n_lstm_layers = 1\n")
            f.write(f"dropout_lstm = 0.2\n")
            f.write(f"d_mlp_head = 10\n")
            f.write(f"dropout_mlp_head = 0.2\n\n")
        
        f.write(f"task = '{task}'\n\n")
        f.write(f"split_by_variant = {split_by_variant}\n")
        f.write(f"regularization = {regularization}\n")
        f.write(f"mixed_variants = {mixed_variants}\n\n")
        if model_type == 'test':
            f.write(f"control_model = False\n\n")
        elif model_type == 'control':
            f.write(f"control_model = True\n\n")

        f.write(f"selected_variants = ['*']\n\n")

        f.write(f"""features = ['Dist AC6/C5,AC6/C4', 'Dist AC6/C5,AC6/C7', 
            'Dist AC6/C4,AC6/C7', 'Dist AC6/C1,AC6/C4',
            'Ang AC6/C1,AC6/C4,AC6/C7', 'Dihe AC6/C5,AC6/C4,AC6/C7,AC6/C9',
            'Dihe AC6/O6,AC6/C4,AC6/C5,AC6/C5-H','Dihe AC6/O8,AC6/C7,AC6/C5,AC6/C5-H', # mode3
            'Dist NDP/H4N2,AC6/C4', 'Dist GLN136/NE2,NDP/O7N', 'Dihe NDP/H1NQ,NDP/C1NQ,NDP/N1N,NDP/C2N', # mode1
            'Ang GLN136/NE2,GLN136/HE22,NDP/O7N', # mode1
            'Dist GLU319/OE1,AC6/C5', 'Ang MG6/M17,AC6/O6,MG6/M16', # mode2
            ]\n\n""")

        f.write("# parallel_folds = True\n")
        f.write("# num_processes = 2\n")

    # write the bash submit file, which runs the job on thor
    shutil.copy(f'/data/karvelis03/dl_kcat/scripts/submit_{model}_1_tmp.sh', submit_filename)
    id_name = str(datetime.now()).replace(' ','').replace(':','').replace('.','')
    dir_name_spec = 'local dir_name=' + id_name + f'{model}_1' + '\n'
    replace_line(submit_filename, 10, dir_name_spec)

    # copy the working script to the subfolder
    shutil.copy(f'/data/karvelis03/dl_kcat/scripts/{model}_1.py', subfolder)

    return subfolder, submit_filename

def main():

    # set the max number of jobs allowed to run at one time
    max_jobs = 10

    if not os.path.exists('./job_wrapper_job_ids.txt'):
        _ = open('./job_wrapper_job_ids.txt', 'a')
        _.close()
    output_text = open('job_wrapper_output.txt', 'w')

    models = ['transformer', 'lstm']
    tasks = ['S/F binary classification', 'NR/R binary classification', 'kcat regression']
    split_by_variants = ['True', 'False']
    regularizations = ['None', "{'l1':0.2}"]
    mixed_variantss = ['False', 'True']
    model_types = ['test', 'control']

    num_jobs = len(models) * len(tasks) * len(split_by_variants) * len(regularizations) *\
               len(mixed_variantss) * len(model_types)

    count = 0
    for model in models:
        for task in tasks:
            for split_by_variant in split_by_variants:
                for regularization in regularizations:
                    for mixed_variants in mixed_variantss:
                        for model_type in model_types:

                            # submit a new job only once we have enough space in the current batch
                            running_jobs = check_jobs()
                            while running_jobs >= max_jobs:
                                time_sleep.sleep(300) # wait a few minutes
                                running_jobs = check_jobs()

                            subfolder, submit_filename = write_config_and_submit_files(model=model,
                                                                                       task=task,
                                                                                       split_by_variant=split_by_variant,
                                                                                       regularization=regularization,
                                                                                       mixed_variants=mixed_variants,
                                                                                       model_type=model_type)
                            print (f"cd {subfolder}; /bt/home/karvelis/scripts/submit.maui.infiniteh_32p.exmem bash {submit_filename} >> {os.getcwd()}/job_wrapper_job_ids.txt")
                            exec = subprocess.Popen(f"cd {subfolder}; /bt/home/karvelis/scripts/submit.maui.infiniteh_32p.exmem bash {submit_filename} >> {os.getcwd()}/job_wrapper_job_ids.txt", shell=True)
                            exec.wait(timeout=None)
                            time_sleep.sleep(20) # wait a few seconds to allow the job time to register on the queue

                            count += 1
                            output_text.write("Submitted %s/%s jobs\n" %(count, num_jobs))
                            output_text.flush()

    output_text.close()


if __name__ == '__main__':
    main()


