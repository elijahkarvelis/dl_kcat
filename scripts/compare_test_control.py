"""
AUTHOR: E Karvelis
PURPOSE:
Quick script that sources from pred_kcat.py to compare between
a control and test model. Run like so: 

>> python compare_test_control.py test_output_text.txt control_output_text.txt (figure.png)

where test_output_text.txt is the output log file written by the ML 
training script (e.g., transformer_1.py) for the test model, 
control_output_text.txt is that for the control model, and the 
optional second argument is the filename to which to save the output
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
import numpy as np


if __name__ == '__main__':
    t_logfile = sys.argv[1]
    c_logfile = sys.argv[2]
    try:
        figname = sys.argv[3]
    except:
        figname = False
    
    t_log = LogFile(t_logfile)
    c_log = LogFile(c_logfile)
    
    # Calculate for each epoch the average error across all folds
    t_avg_rmse = t_log.epochs.groupby(by='Epoch').mean()
    c_avg_rmse = c_log.epochs.groupby(by='Epoch').mean()
    t_std_rmse = t_log.epochs.groupby(by='Epoch').std()
    c_std_rmse = c_log.epochs.groupby(by='Epoch').std()

    # Tabulate the number of folds
    c_cvfolds = np.unique(c_log.epochs['CV fold'].to_numpy()).shape[0]
    t_cvfolds = np.unique(t_log.epochs['CV fold'].to_numpy()).shape[0]
    
    # Plot
    fig, ax = plt.subplots(1,1,figsize=None)

    epochs = t_avg_rmse.index.values
    t_rmse = t_avg_rmse['RMSE'].to_numpy()
    c_rmse = c_avg_rmse['RMSE'].to_numpy()
    t_rmse_err = t_std_rmse['RMSE'].to_numpy() / (t_cvfolds**0.5)
    c_rmse_err = c_std_rmse['RMSE'].to_numpy() / (c_cvfolds**0.5)

    # plot the average profiles
    ax.plot(epochs, t_rmse, label='Test', color='b', alpha=0.8)
    ax.plot(epochs, c_rmse, label='Control', color='k', alpha=0.8)

    # Shade in +/- 1 SEM
    ax.fill_between(epochs, t_rmse-t_rmse_err, t_rmse+t_rmse_err, color='b', alpha=0.3)
    ax.fill_between(epochs, c_rmse-c_rmse_err, c_rmse+c_rmse_err, color='k', alpha=0.3)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('RMSE (validation)')
    ax.set_ylim((0.9,3))
    ax.legend()

    plt.show()

    fig.tight_layout()
    if figname:
        fig.savefig(figname, dpi=300)