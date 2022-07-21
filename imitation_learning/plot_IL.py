from __future__ import print_function
import matplotlib.pyplot as plt
import sys
import torch

sys.path.append("../")

from datetime import datetime
import numpy as np
import gym
import os
import json

from agent.bc_agent import BCAgent
from utils import *
import tensorflow as tf
import seaborn as sns
import pandas as pd


def rolling_avg(numbers, window_size):

    numbers_series = pd.Series(numbers)
    windows = numbers_series.rolling(window_size)
    moving_averages = windows.mean()

    moving_averages_list = moving_averages.tolist()
    without_nans = moving_averages_list[window_size - 1:]

    return without_nans

if __name__ == "__main__":
    import os
    file1 = None
    file2 = None
    file3 = None
    dir_path = os.path.dirname(os.path.realpath(__file__))
    basepath = dir_path+'/tensorboard/IL_BS_64_HL_1'
    for entry in os.listdir(basepath):
        if os.path.isfile(os.path.join(basepath, entry)):
            print(entry)
            file1 = tf.train.summary_iterator(
                dir_path + "/tensorboard/IL_BS_64_HL_1/"+entry)

    basepath = dir_path + '/tensorboard/IL_BS_64_HL_3'
    for entry in os.listdir(basepath):
        if os.path.isfile(os.path.join(basepath, entry)):
            print(entry)
            file2 = tf.train.summary_iterator(
                dir_path + "/tensorboard/IL_BS_64_HL_3/" + entry)

    basepath = dir_path + '/tensorboard/IL_BS_64_HL_5'
    for entry in os.listdir(basepath):
        if os.path.isfile(os.path.join(basepath, entry)):
            print(entry)
            file3 = tf.train.summary_iterator(
                dir_path + "/tensorboard/IL_BS_64_HL_5/" + entry)


    #file1 = tf.train.summary_iterator(dir_path+"/tensorboard/IL_BS-65_HL-0/events.out.tfevents.1623100452.tfpool20")
    #file2 = tf.train.summary_iterator(dir_path+"/tensorboard/IL_BS-65_HL-1/events.out.tfevents.1623102203.tfpool20")
    #file3 = tf.train.summary_iterator(dir_path+"/tensorboard/IL_BS-65_HL-3/events.out.tfevents.1623106485.tfpool20")
    # for summary in ok:
    #     print(summary)

    ta = []
    va = []
    ta2 = []
    va2 = []
    ta3 = []
    va3 = []
    for e in file1:
        for v in e.summary.value:
            if v.tag == 'TrainingAccuracy_1':
                print("Training Acc "+str(v.simple_value))
                ta.append(v.simple_value)
            if v.tag == 'ValidationAccuracy_1':
                print("Validation Acc "+str(v.simple_value))
                va.append(v.simple_value)
        print('************************1************************')


    for e in file2:
        for v in e.summary.value:
            if v.tag == 'TrainingAccuracy_1':
                print("Training Acc "+str(v.simple_value))
                ta2.append(v.simple_value)
            if v.tag == 'ValidationAccuracy_1':
                print("Validation Acc "+str(v.simple_value))
                va2.append(v.simple_value)
        print('************************2************************')


    for e in file3:
        for v in e.summary.value:
            if v.tag == 'TrainingAccuracy_1':
                print("Training Acc "+str(v.simple_value))
                ta3.append(v.simple_value)
            if v.tag == 'ValidationAccuracy_1':
                print("Validation Acc "+str(v.simple_value))
                va3.append(v.simple_value)
        print('***********************3*************************')

    window_size = 40
    # Calculate rolling average
    va_roll = rolling_avg(va, window_size)
    va2_roll = rolling_avg(va2, window_size)
    va3_roll = rolling_avg(va3, window_size)

    sns.set_theme(style="darkgrid")
    axt = sns.lineplot(np.arange(1000), ta, legend='brief', label="HL 1")
    axt = sns.lineplot(np.arange(1000), ta2, legend='brief', label="HL 3")
    axt = sns.lineplot(np.arange(1000), ta3, legend='brief', label="HL 5")
    axt.set(xlabel='Number of Minibatches', ylabel='Training Accuracy')
    axt.set(title='Imitation Learning for CarRacing : Training Trend')
    #plt.show()
    plt.savefig("trainAcc.png")
    plt.close()
    axv = sns.lineplot(np.arange(len(va_roll)), va_roll, legend='brief', label="HL 1")
    axv = sns.lineplot(np.arange(len(va2_roll)), va2_roll, legend='brief', label="HL 3")
    axv = sns.lineplot(np.arange(len(va3_roll)), va3_roll, legend='brief', label="HL 5")
    axv.set(xlabel='Number of Minibatches', ylabel='Validation Accuracy')
    axv.set(title='Imitation Learning for CarRacing : Validation Trend')
    #plt.show()
    plt.savefig("valAcc.png")
    print('... finished')