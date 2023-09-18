"""
AUTHOR: E Karvelis
PURPOSE:
Quick script that sources from pred_kcat.py to compare between
a control and test model. Run like so: 

>> python compare_test_control.py test_output_text.txt control_output_text.txt (figure.png)

where test_output_text.txt is the output log file written by the ML 
training script (e.g., transformer_1.py) for the test model, 
control_output_text.txt is that for the control model, and the 
optional third and fourth arguments are the metric to plot and 
the filename to which to save the output figure.
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


def rmse_plot(t_log, c_log, figname=False):
    # INPUT:
    # t_log -- a LogFile object for the test job's output file
    # c_log -- a LogFile object for the control job's output file
    # figname -- name of file to which to save plot
    # OUTPUT:
    # Plots the average RMSE across CV folds, shaded by +/- 1SEM, as
    # a function of number of training epochs. Saves the plot to 
    # figname if specified.

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
    
    return


def accuracy_plot(t_log, c_log, figname=False):
    # INPUT:
    # t_log -- a LogFile object for the test job's output file
    # c_log -- a LogFile object for the control job's output file
    # figname -- name of file to which to save plot
    # OUTPUT:
    # Plots the average accuracy across CV folds, shaded by +/- 1SEM, 
    # as a function of number of training epochs. Saves the plot to 
    # figname if specified.

    # Calculate for each epoch the average accuracy across all folds
    t_avg_acc = t_log.epochs.groupby(by='Epoch').mean()
    c_avg_acc = c_log.epochs.groupby(by='Epoch').mean()
    t_std_acc = t_log.epochs.groupby(by='Epoch').std()
    c_std_acc = c_log.epochs.groupby(by='Epoch').std()

    # Tabulate the number of folds
    c_cvfolds = np.unique(c_log.epochs['CV fold'].to_numpy()).shape[0]
    t_cvfolds = np.unique(t_log.epochs['CV fold'].to_numpy()).shape[0]
    
    # Plot
    fig, ax = plt.subplots(1,1,figsize=None)

    epochs = t_avg_acc.index.values
    t_acc = t_avg_acc['Acc.'].to_numpy()
    c_acc = c_avg_acc['Acc.'].to_numpy()
    t_acc_err = t_std_acc['Acc.'].to_numpy() / (t_cvfolds**0.5)
    c_acc_err = c_std_acc['Acc.'].to_numpy() / (c_cvfolds**0.5)

    # plot the average profiles
    ax.plot(epochs, t_acc, label='Test', color='b', alpha=0.8)
    ax.plot(epochs, c_acc, label='Control', color='k', alpha=0.8)

    # Shade in +/- 1 SEM
    ax.fill_between(epochs, t_acc-t_acc_err, t_acc+t_acc_err, color='b', alpha=0.3)
    ax.fill_between(epochs, c_acc-c_acc_err, c_acc+c_acc_err, color='k', alpha=0.3)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (validation)')
    # ax.set_ylim((0.45,1))
    ax.legend()

    plt.show()

    fig.tight_layout()
    if figname:
        fig.savefig(figname, dpi=300)
    
    return


def auroc_plot(t_log, c_log, figname=False):
    # INPUT:
    # t_log -- a LogFile object for the test job's output file
    # c_log -- a LogFile object for the control job's output file
    # figname -- name of file to which to save plot
    # OUTPUT:
    # Plots the average AUROC across CV folds, shaded by +/- 1SEM, 
    # as a function of number of training epochs. Saves the plot to 
    # figname if specified.

    # Calculate for each epoch the average accuracy across all folds
    t_avg_auroc = t_log.epochs.groupby(by='Epoch').mean()
    c_avg_auroc = c_log.epochs.groupby(by='Epoch').mean()
    t_std_auroc = t_log.epochs.groupby(by='Epoch').std()
    c_std_auroc = c_log.epochs.groupby(by='Epoch').std()

    # Tabulate the number of folds
    c_cvfolds = np.unique(c_log.epochs['CV fold'].to_numpy()).shape[0]
    t_cvfolds = np.unique(t_log.epochs['CV fold'].to_numpy()).shape[0]
    
    # Plot
    fig, ax = plt.subplots(1,1,figsize=None)

    epochs = t_avg_auroc.index.values
    t_auroc = t_avg_auroc['AUROC'].to_numpy()
    c_auroc = c_avg_auroc['AUROC'].to_numpy()
    t_auroc_err = t_std_auroc['AUROC'].to_numpy() / (t_cvfolds**0.5)
    c_auroc_err = c_std_auroc['AUROC'].to_numpy() / (c_cvfolds**0.5)

    # plot the average profiles
    ax.plot(epochs, t_auroc, label='Test', color='b', alpha=0.8)
    ax.plot(epochs, c_auroc, label='Control', color='k', alpha=0.8)

    # Shade in +/- 1 SEM
    ax.fill_between(epochs, t_auroc-t_auroc_err, t_auroc+t_auroc_err, color='b', alpha=0.3)
    ax.fill_between(epochs, c_auroc-c_auroc_err, c_auroc+c_auroc_err, color='k', alpha=0.3)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('AUROC (validation)')
    # ax.set_ylim((0.45,1))
    ax.legend()

    plt.show()

    fig.tight_layout()
    if figname:
        fig.savefig(figname, dpi=300)
    
    return



if __name__ == '__main__':
    t_logfile = sys.argv[1]
    c_logfile = sys.argv[2]
    metric = sys.argv[3]
    try:
        figname = sys.argv[4]
    except:
        figname = False
    
    t_log = LogFile(t_logfile)
    c_log = LogFile(c_logfile)
    
    if metric.upper() == 'RMSE':
        rmse_plot(t_log, c_log, figname)
    elif metric.upper() == 'ACCURACY':
        accuracy_plot(t_log, c_log, figname)
    elif metric.upper() == 'AUROC':
        auroc_plot(t_log, c_log, figname)