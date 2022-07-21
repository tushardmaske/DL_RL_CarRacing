from __future__ import print_function
import matplotlib.pyplot as plt
import sys
import json
import pandas as pd

sys.path.append("../")

from utils import *
import tensorflow as tf
import seaborn as sns

def rolling_avg(numbers, window_size):

    numbers_series = pd.Series(numbers)
    windows = numbers_series.rolling(window_size)
    moving_averages = windows.mean()

    moving_averages_list = moving_averages.tolist()
    without_nans = moving_averages_list[window_size - 1:]

    return without_nans

if __name__ == "__main__":
    import os
    dir_path = os.path.dirname(os.path.realpath(__file__))

    basepath = dir_path+'/tensorboard/train/CartPole'

    for entry in os.listdir(basepath):
        if os.path.isfile(os.path.join(basepath, entry)):
            print(entry)
            file1 = tf.train.summary_iterator(
                dir_path + "/tensorboard/train/CartPole/"+entry)



    #file1 = tf.train.summary_iterator(dir_path+"/tensorboard/IL_BS-65_HL-0/events.out.tfevents.1623100452.tfpool20")
    #file2 = tf.train.summary_iterator(dir_path+"/tensorboard/IL_BS-65_HL-1/events.out.tfevents.1623102203.tfpool20")
    #file3 = tf.train.summary_iterator(dir_path+"/tensorboard/IL_BS-65_HL-3/events.out.tfevents.1623106485.tfpool20")
    # for summary in ok:
    #     print(summary)

    rw = []

    for e in file1:
        for v in e.summary.value:
            if v.tag == 'episode_reward_1':
                print("Training Acc "+str(v.simple_value))
                rw.append(v.simple_value)

        print('************************1************************')

    window_size = 50
    # Calculate moving average
    rw_roll = rolling_avg(rw, window_size)

    sns.set_theme(style="darkgrid")
    axtrain = sns.lineplot(np.arange(len(rw_roll)), rw_roll)
    axtrain.set(xlabel='No of Episodes', ylabel='Achieved Rewards')
    axtrain.set(title='Training results of CartPole : Episode Rewards')
    #axt.set(ylim=(0, 200))
    #plt.show()
    plt.savefig("CP_TrainingRewards.png")

    plt.close()


    # Opening JSON file
    f = open('results/cartpole_results_dqn-20210608-074130.json', )

    # returns JSON object as
    # a dictionary
    data = json.load(f)

    # Iterating through the json
    # list
    resultRewards1 = []
    resultRewards2 = []
    resultRewards3 = []
    for i in data['episode_rewards']:
        print(i)
        resultRewards1.append(i)
    # Closing file
    f.close()

    sns.set_theme(style="darkgrid")
    axRes = sns.lineplot(np.arange(15), resultRewards1)
    axRes.set(xlabel='No of Episodes', ylabel='Achieved Rewards')
    axRes.set(title='Validation results of CarRacing : Episode Rewards')
    #axt.set(ylim=(0, 200))
    #axRes.set(ylim=(500, 1100))
    #plt.show()
    plt.savefig("CP_TestingRewards.png")


    print('... finished')
