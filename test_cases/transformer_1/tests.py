# Import packages
import subprocess
from glob import glob
import shutil
import time
import os

# Hardcoded settings
submit_script = 'submit.maui.infiniteh_8p.exmem'

if not os.path.exists('./transformer_1.py'):
    raise ValueError('./transformer_1.py missing -- this file is needed to submit tests.')

dirs = glob('./class*') + glob('./reg*')

for dir in dirs:
    # copy the working script to the job location
    shutil.copy('./transformer_1.py', dir)

    print (f"cd {dir}; ../{submit_script} bash submit_transformer_1.sh >> ../job_ids.txt")
    exec = subprocess.Popen(f"cd {dir}; ../{submit_script} bash submit_transformer_1.sh >> ../job_ids.txt", shell=True)
    exec.wait(timeout=None)
    time.sleep(5) # pause a few seconds between submissions


