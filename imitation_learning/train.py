from __future__ import print_function

import sys
sys.path.append("../") 
import torch
import pickle
import numpy as np
import os
import gzip
import matplotlib.pyplot as plt
import random
from utils import *
from imitation_learning.agent.bc_agent import BCAgent
from tensorboard_evaluation import Evaluation

def accuracy(y: np.ndarray, predictions: np.ndarray) -> float:
    """Calculate accuracy given the ground truths and predictions.

    Args:
        y: Target labels, not one-hot encoded.
        predictions: Predictions of the model.

    Returns:
        Accuracy.

    """
    y_predicted = np.argmax(predictions, axis=-1)
    return np.sum(np.equal(y_predicted, y)) / len(y)

def validation(X_valid, y_valid, agent) -> float:
    predictions = []
    batch_size = 20
    # agent = BCAgent()
    # calculate how many minibatches we get out of the validation dataset given the batch size
    num_val_batches = np.floor(len(X_valid) // batch_size)
    num_val_batches = int(num_val_batches)
    #    assert len(X_valid) % batch_size == 0, (
    #   f"Training dataset size of {len(X_valid)} is not divisible by batch size {batch_size}.")

    for batch_num in range(num_val_batches):
        # get the minibatch data given the current minibatch number
        minibatch_start = batch_num * batch_size
        minibatch_end = (batch_num + 1) * batch_size
        x_batch = X_valid[minibatch_start:minibatch_end]

        y_predicted = agent.predict(x_batch).cpu().detach().numpy()
        predictions.append(y_predicted)

    # concatenate the minibatch validation predictions back together and calculate the accuracy
    predictions = np.concatenate(predictions, axis=0)
    y_valid = y_valid[0:batch_size * num_val_batches]
    eval_accuracy = accuracy(y_valid, predictions)

    # return accuracy and loss for this epoch
    return eval_accuracy


def read_data(datasets_dir="./data", frac = 0.1):
    """
    This method reads the states and actions recorded in drive_manually.py 
    and splits it into training/ validation set.
    """
    print("... read data")
    data_file = os.path.join(datasets_dir, 'data.pkl.gzip')
  
    f = gzip.open(data_file, 'rb')
    data = pickle.load(f)

    # get images as features and actions as targets
    X = np.array(data["state"]).astype('float32')
    y = np.array(data["action"]).astype('float32')

    # split data into training and validation set
    n_samples = len(data["state"])
    X_train, y_train = X[:int((1-frac) * n_samples)], y[:int((1-frac) * n_samples)]
    X_valid, y_valid = X[int((1-frac) * n_samples):], y[int((1-frac) * n_samples):]
    return X_train, y_train, X_valid, y_valid


def preprocessing(X_train, y_train, X_valid, y_valid, history_length=1):

    # preprocess your data here.
    # 1. convert the images in X_train/X_valid to gray scale. If you use rgb2gray() from utils.py, the output shape (96, 96, 1)
    # 2. you can train your model with discrete actions (as you get them from read_data) by discretizing the action space 
    #    using action_to_id() from utils.py.
    X_train = rgb2gray(X_train)
    X_valid = rgb2gray(X_valid)

    # To view the image
    # arr = np.asarray(X_train[20])
    # plt.figure()
    # plt.imshow(arr)
    # plt.show()

    # History:
    # At first you should only use the current image as input to your network to learn the next action. Then the input states
    # have shape (96, 96, 1). Later, add a history of the last N images to your state so that a state has shape (96, 96, N).

    x_train = torch.tensor(X_train)
    x_train_hist = torch.tensor((), dtype=torch.int32)
    x_train_hist = x_train_hist.new_empty(((len(x_train) - history_length), history_length + 1, 96, 96))
    for i in range(len(x_train) - history_length):
        for j in range(history_length + 1):
            x_train_hist[i][j] = x_train[i + history_length - j]

    y_train_hist = y_train[history_length:len(x_train)]

    x_valid = torch.tensor(X_valid)
    x_valid_hist = torch.tensor((), dtype=torch.int32)
    x_valid_hist = x_valid_hist.new_empty(((len(x_valid) - history_length), history_length + 1, 96, 96))
    #  print(history_length)
    for i in range(len(x_valid) - history_length):
        for j in range(history_length + 1):
            x_valid_hist[i][j] = x_valid[i + history_length - j]

    y_valid_hist = y_valid[history_length:len(x_valid)]

    y_train_discrete = []
    y_valid_discrete = []

    for i in range(len(y_train_hist)):
        y_train_discrete.append(action_to_id(y_train_hist[i]))
    for i in range(len(y_valid_hist)):
        y_valid_discrete.append(action_to_id(y_valid_hist[i]))

    return x_train_hist, np.array(y_train_discrete), x_valid_hist, np.array(y_valid_discrete)

def train_model(X_train, y_train, X_valid, y_valid, n_minibatches, history_length=0, batch_size=64, lr=1e-4, model_dir="./models", tensorboard_dir="./tensorboard"):
    
    # create result and model folders
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)  
 
    print("... train model")

    # specify your agent with the neural network in agents/bc_agent.py
    agent = BCAgent(history_length=history_length)
    name = "IL_BS_" + str(batch_size) + "_HL_" + str(history_length + 1) + "_"
    tensorboard_eval = Evaluation(tensorboard_dir, name=name, stats=["TrainingAccuracy", "ValidationAccuracy"])

    # implement the training
    # 1. write a method sample_minibatch and perform an update step
    # 2. compute training/ validation accuracy and loss for the batch and visualize them with tensorboard. You can watch the progress of
    #    your training *during* the training in your web browser

    # training loop
    y_ind = []
    y_train_preds = []
    y_over_n_minibatches = []
    for j in range(5):
        y_ind.append([i for i, x in enumerate(y_train) if x == j])

    best_validation_acc = 0
    for i in range(n_minibatches):
        y_randInd = []
        num = np.floor(batch_size / 5)
        for k in range(int(num)):
            for j in y_ind:
                a = random.sample(j, 1)
                y_randInd = y_randInd + a
        rem = batch_size - num * 5
        for k in range(int(rem)):
            h = int(np.random.choice(5, 1))
            b = y_ind[h]
            a = random.sample(b, 1)
            y_randInd = y_randInd + a

        x_batch = X_train[y_randInd]
        y_batch = y_train[y_randInd]
        train_idx = np.arange(len(x_batch))
        # for i in range(len(x_batch)):
        np.random.shuffle(train_idx)
        x_train_shuffled = x_batch[train_idx]
        y_train_shuffed = y_batch[train_idx]

        x_train_shuffled = x_train_shuffled.float()

        loss = agent.update(x_train_shuffled, y_train_shuffed)
        y_label = agent.predict(x_batch).cpu()
        y_label = y_label.detach().numpy()
        y_train_preds.append(y_label)
        y_over_n_minibatches.append(y_batch)
        # print("Loss of batch {} is {} ".format(i, loss))

        # compute training/ validation accuracy and write it to tensorboard
        train_accuracy = 0
        validation_accuracy = 0
        if (i+1) % 10 == 0:
            train_accuracy = accuracy(np.concatenate(y_over_n_minibatches, axis=0), np.concatenate(y_train_preds, axis=0))
            validation_accuracy = validation(X_valid, y_valid, agent)
            print("Training accuracy after {} mini batches is {}".format(i+1, train_accuracy))
            print("Validation accuracy after {} mini batches is {}".format(i+1, validation_accuracy))
            eval_dict = {"TrainingAccuracy": train_accuracy, "ValidationAccuracy": validation_accuracy}
            tensorboard_eval.write_episode_data(i, eval_dict)

    # save your agent
            # if best_validation_acc < validation_accuracy:
                # model_dir = agent.save(os.path.join(model_dir, "agent.pt"))
    agent.save(os.path.join(model_dir, "agent.pt"))
                # best_validation_acc = validation_accuracy
                # print("-------------- Model saved at validation accuracy {} --------------".format(best_validation_acc))


if __name__ == "__main__":

    hist_length = 4
    # read data    
    X_train, y_train, X_valid, y_valid = read_data("./data")

    # preprocess data
    X_train, y_train, X_valid, y_valid = preprocessing(X_train, y_train, X_valid, y_valid, history_length=hist_length)
    # train model (you can change the parameters!)
    train_model(X_train, y_train, X_valid, y_valid, n_minibatches=10000, batch_size=64, lr=1e-4, history_length=hist_length)
 
