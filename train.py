import argparse
import os
import pandas as pd
import torch
from random import shuffle
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy import sparse
import torch.nn as nn
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
from collections import  defaultdict
from os import listdir
import itertools
from torch.utils.data.dataset import Dataset
from model import sakt
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

def get_data(data_path,max_sequence_size):
    
    array_y = []
    array_problem = []


    #Loading the data in a pandas DataFrame
    df = pd.read_csv(data_path)
    users_data = df.groupby('student_id')[['problem_id','condition','skill_name','correct']].agg(lambda x:list(x))

    index_to_id = np.unique([*itertools.chain.from_iterable(users_data['problem_id'])])
    id_to_index = {index_to_id[i]: i for i in range(len(index_to_id))}

    for index,student in users_data.iterrows():
        sequence_size = len(student['problem_id'])
        student_problem_id = [id_to_index[i] for i in student['problem_id']]
        if sequence_size > max_sequence_size:
            for i in range(max_sequence_size,sequence_size):
                array_y.append(student['correct'][(i - max_sequence_size):i])
                array_problem.append(student_problem_id[(i - max_sequence_size):i])
        else:
            array_y.append(student['correct'] + [-1] * (max_sequence_size - sequence_size))
            array_problem.append(student_problem_id + [0] * (max_sequence_size - sequence_size))


    y = np.array(array_y)
    problem = np.array(array_problem)
    
    return y,problem

def train():
    """ 
    Train SAKT model
    
    Arguments: 
        train_data 
        val_data    
    """

    criterion = nn.CrossEntropyLoss()


def get_args():
        
    parser = argparse.ArgumentParser(description='Train SAKT')
    parser.add_argument('--dataset','-dt', type=str,default=Path('./data/'))
    parser.add_argument('--max_sequence_size','-ms', type=int, default=400)
    parser.add_argument('--embed_size','-em', type=int, default=200)
    parser.add_argument('--drop_prob','-d', type=float, default=0.2)
    parser.add_argument('--batch_size','-b', type=int, default=200)
    parser.add_argument('--learning-rate','-lr', type=float, default=1e-3)
    parser.add_argument('--epochs','-e', type=int, default=300)
    
    return parser.parse_args()




if __name__ == "__main__":

    
    args = get_args()
    
    data_path = './data/' + os.listdir(Path('./data/'))[0]
    y,problem = get_data(data_path,args.max_sequence_size)

    print(y)
    print(problem)
    
    # The parameters are learned by minimizing the cross entropy loss between pt and rt
    # pt = output of prediction layer
    # rt = correctness of the student's answer
    # The performance is compared using the Area Under Curve (AUC)
    # ADAM optimizer with lr of 0.001
    # dropout rate = 0.2
    #We set the maximum length of the sequence, n as roughly proportional to the average exercise tags per student.
    
    #TODO
    # check if divide the sequence with more than the max_sequence_size will effect when predict for a specific student
    #After,understand how the data will enter in the network and check the tensors shape

    #DONE
    #Look in kaggle, how to preprocess student id's
    #Start to think how to preprocess the data and try to model the train and test data




