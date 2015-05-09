import csv
import numpy as np
import pandas as pd
import math
from cgen_include import *

### Functions ###

def loadData(file_name, taxas):
    '''
    Given a file, and a list of taxa levels desired, create a dataframe with the sample id, disease status and feature columns. No lineage was provided for this data (and it was only provided at the order/family level, so some munging needed to happen to make the taxa levels consistent and provide phyla as well as orders).
    '''
    data = []
    line_count = 0
    f = open(file_name)
    contents = csv.reader(f, delimiter=',')
    feature_dict = {'order': {'Erysipelotrichaceae': 'Erysiopelotrichales', 'Neisseriaceae': 'Neisseriales', 'Clostridiales': 'Clostridiales', 'Pasteurellaceae': 'Pasteurellales', 'Bifidobacteriaceae': 'Bifidobacteriales', 'Fusobacteriaceae': 'Fusobacteriales', 'Veillonellaceae': 'Selenomonadales', 'Enterobacteriaceae': 'Enterobacteriales', 'Micrococcaceae': 'Actinomycetales', 'Verrucomicrobiaceae': 'Verrucomicrobiales', 'Gemellaceae': 'Gemellales', 'Coriobacteriaceae': 'Coriobacteriales', 'Bacteroidales': 'Bacteroidales'}, 'phylum': {'Erysipelotrichaceae': 'Firmicutes', 'Neisseriaceae': 'Proteobacteria', 'Clostridiales': 'Firmicutes', 'Pasteurellaceae': 'Proteobacteria', 'Bifidobacteriaceae': 'Acinobacteria', 'Fusobacteriaceae': 'Fusobacteriales', 'Veillonellaceae': 'Firmicutes', 'Enterobacteriaceae': 'Proteobacteria', 'Micrococcaceae': 'Acinobacteria', 'Verrucomicrobiaceae': 'Verrucomicrobia', 'Gemellaceae': 'Firmicutes', 'Coriobacteriaceae': 'Acinobacteria', 'Bacteroidales': 'Bacteroidetes'}}
    feature_map = {}
    data_columns = ['Sample', 'Diagnosis']
    for level in taxas:
        for item in feature_dict[level]:
            data_columns.append(feature_dict[level][item]+'_'+level)
            feature_map[feature_dict[level][item]+'_'+level] = item
    #Loop through each line of the file
    ix = 1
    for line in contents:
        if ix == 1:
            headers = line
        else:
            #Put the data into dict with header as label
            ln_dict = {} 
            for col in data_columns:
                #bacterial percentages
                if col == 'Diagnosis':
                    if line[headers.index(col)] == 'CD':
                        ln_dict[col] = 1
                    else:
                        ln_dict[col] = -1
                elif col == 'Sample':
                    ln_dict[col] = line[headers.index('subject')]
                else:
                    col_ix = headers.index(feature_map[col])
                    if col in ln_dict:
                        ln_dict[col] = ln_dict[col] + float(line[col_ix])
                    else:
                        ln_dict[col] = float(line[col_ix])
            data.append(ln_dict)
        ix += 1
    f.close()
    #Turn the arrays of dicts into data frames and return them
    return pd.DataFrame(data)

def createSplit(df, type):
    '''
    Given a dataframe, this splits the data so that there are an even number of positive and negative examples in the training set.
    '''
    diagnosis_counts = df['Diagnosis'].value_counts()
    cd_arr = []
    cd_df = df[df['Diagnosis'] == 1]
    no_df = df[df['Diagnosis'] == -1]
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
    '''
    This splits a dataframe into two, one with the features and one with the classes (ground truth)
    '''
    return df[features], df[class_col]
 
def getOrigSplitData(random_seed, taxas):
    '''
    The "public" interface to this file, what the other files call in order to get the data for this dataset. The taxa levels desired are required.
    '''
    file = 'samples_1040.csv'
    df = loadData(file, taxas)
    target_col = ['Diagnosis']
    feature_cols = list(df.columns.values)
    feature_cols.remove('Sample')
    feature_cols.remove('Diagnosis')
    
    #This dataset is very skewed (not evenly split between disease and health for diagnosis). I didn't want to rely on the standard train_test_split function and so wrote a custom function to do the split for this data.
    test_df, train_df = createSplit(df, 'test')
    valid_df, train_df = createSplit(train_df, 'valid')
    train_features, train_classes = splitFeaturesClasses(train_df, feature_cols, target_col)
    valid_features, valid_classes = splitFeaturesClasses(valid_df, feature_cols, target_col)
    test_features, test_classes = splitFeaturesClasses(test_df, feature_cols, target_col)
    
    return {'train': {'features': train_features, 'classes': train_classes, 'feature_cols': feature_cols}, 'valid': {'features': valid_features, 'classes': valid_classes, 'feature_cols': feature_cols}, 'test': {'features': test_features, 'classes': test_classes, 'feature_cols': feature_cols}}
    