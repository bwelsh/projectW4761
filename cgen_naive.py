import sys
import json
import random
import csv
import numpy as np
import pandas as pd
import math
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from cgen_include import *

###Functions###

def loadData(file_name):
    """
    TODO
    """
    data = []
    #headers = ['x', 'y', 'label']
    line_count = 0
    f = open(file_name)
    contents = csv.reader(f, delimiter=',')
    #Loop through each line of the file
    ix = 1
    for line in contents:
        if ix == 1:
            headers = line
        else:
            #Put the data into dict with header as label
            ln_dict = {} 
            for i in range(0, len(headers)):
                #bacterial percentages
                if i > 6:
                    ln_dict[headers[i]] = float(line[i])
                elif headers[i] == 'Diagnosis':
                    if line[i] == 'CD':
                        ln_dict[headers[i]] = 1
                    else:
                        ln_dict[headers[i]] = -1
                else:
                    ln_dict[headers[i]] = line[i]
            data.append(ln_dict)
        ix += 1
    f.close()
    #Turn the arrays of dicts into data frames and return them
    return pd.DataFrame(data)

def createSplit(df, type):
    diagnosis_counts = df['Diagnosis'].value_counts()
    cd_arr = []
    cd_df = df[df['Diagnosis'] == 1]
    no_df = df[df['Diagnosis'] == -1]
    # FIX THIS, balancing the dataset
    if type == 'test':
        if diagnosis_counts[1] > diagnosis_counts[-1]:
            diff = diagnosis_counts[1] - diagnosis_counts[-1]
            cd_arr = random.sample(cd_df.index.tolist(), diff+math.floor(diagnosis_counts[-1]*0.2))
    if type == 'valid':
        cd_arr = random.sample(cd_df.index.tolist(), math.floor(diagnosis_counts[1]*0.2))
    no_arr = random.sample(no_df.index.tolist(), math.floor(diagnosis_counts[-1]*0.2))
    cd_other = cd_df[cd_df.index.isin(cd_arr)]
    cd_train = cd_df[~cd_df.index.isin(cd_arr)]
    no_other = no_df[no_df.index.isin(no_arr)]
    no_train = no_df[~no_df.index.isin(no_arr)]
    other_df = cd_other.append(no_other, ignore_index=True)
    train_df = cd_train.append(no_train, ignore_index=True)
    return other_df, train_df
    
def splitFeaturesClasses(df, features, class_col):
    return df[features], df[class_col]
    
###Main###
file = 'samples_1040.csv'
random_seed = 4
random.seed(random_seed)
feature_cols = ['Erysipelotrichaceae', 'Neisseriaceae', 'Clostridiales', 'Pasteurellaceae', 'Bifidobacteriaceae', 'Fusobacteriaceae', 'Veillonellaceae', 'Enterobacteriaceae', 'Micrococcaceae', 'Verrucomicrobiaceae', 'Gemellaceae', 'Coriobacteriaceae', 'Bacteroidales']
target_col = ['Diagnosis']
df = loadData(file)
test_df, train_df = createSplit(df, 'test')
valid_df, train_df = createSplit(train_df, 'valid')
train_features, train_classes = splitFeaturesClasses(train_df, feature_cols, target_col)
valid_features, valid_classes = splitFeaturesClasses(valid_df, feature_cols, target_col)
test_features, test_classes = splitFeaturesClasses(test_df, feature_cols, target_col)

#Scale data to same scale, using train data to find the scale
head_cols = train_features.columns
scaler = preprocessing.StandardScaler().fit(train_features)
scaled_train = scaler.transform(train_features)
scaled_train = pd.DataFrame(scaled_train)
scaled_train.columns = head_cols
scaled_valid = scaler.transform(valid_features)
scaled_valid = pd.DataFrame(scaled_valid)
scaled_valid.columns = head_cols
scaled_test = scaler.transform(test_features)
scaled_test = pd.DataFrame(scaled_test)
scaled_test.columns = head_cols
        
feature_class_split = {'train': {'features': scaled_train, 'classes': train_classes}, 'validate': {'features': scaled_valid, 'classes': valid_classes}, 'test': {'features': scaled_test, 'classes': test_classes}}

#print (feature_class_split['test']['features'][:1])
#print (feature_class_split['test']['classes'][:1])