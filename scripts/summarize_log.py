"""
AUTHOR: E Karvelis
PURPOSE:
Quick script that sources from LogFile class in pred_kcat.py to summarize 
model training progress. Run like so: 

>> python summarize_log.py output_text.txt (figure.png)

where output_text.txt is the output log file written by the ML 
training script (e.g., transformer_1.py), and the optional 
second argument is the filename to which to save the output
figure
"""

# Import LogFile class
import sys
sys.path.append('/data/karvelis03/dl_kcat/scripts/')
from pred_kcat import LogFile

# Import other dependencies
import pandas as pd
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == '__main__':

	logfile = sys.argv[1]
	try:
		figname = sys.argv[2]
	except:
		figname = False

	log = LogFile(logfile, figname=figname)
	log.plot_summary()


