import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn import cross_validation as crossv
from sklearn import linear_model, metrics
from sklearn import tree, naive_bayes
from sklearn import ensemble
from sklearn import neighbors
from sklearn import cluster
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn import grid_search
from sklearn import preprocessing, feature_selection
import random
import csv
import math
import sklearn

### Functions ###

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
    
def addPhyla(feature_data):
    for data in feature_data:
        feature_data[data]['Firmicutes'] = feature_data[data]['Erysipelotrichaceae'] + feature_data[data]['Veillonellaceae'] + feature_data[data]['Clostridiales'] + feature_data[data]['Gemellaceae'] 
        feature_data[data]['Proteobacteria'] = feature_data[data]['Neisseriaceae'] + feature_data[data]['Enterobacteriaceae'] + feature_data[data]['Pasteurellaceae']
        feature_data[data]['Acinobacteria'] = feature_data[data]['Micrococcaceae'] + feature_data[data]['Coriobacteriaceae'] + feature_data[data]['Bifidobacteriaceae']
    return feature_data
    
def getOrigData(random_seed, with_poly, with_phyla):
    file = '/home/barbara/Documents/genomics/project/samples_1040.csv'
    feature_cols = ['Erysipelotrichaceae', 'Neisseriaceae', 'Clostridiales', 'Pasteurellaceae', 'Bifidobacteriaceae', 'Fusobacteriaceae', 'Veillonellaceae', 'Enterobacteriaceae', 'Micrococcaceae', 'Verrucomicrobiaceae', 'Gemellaceae', 'Coriobacteriaceae', 'Bacteroidales']
    target_col = ['Diagnosis']
    df = loadData(file)
    test_df, train_df = createSplit(df, 'test')
    valid_df, train_df = createSplit(train_df, 'valid')
    train_features, train_classes = splitFeaturesClasses(train_df, feature_cols, target_col)
    valid_features, valid_classes = splitFeaturesClasses(valid_df, feature_cols, target_col)
    test_features, test_classes = splitFeaturesClasses(test_df, feature_cols, target_col)
    
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
    
    feature_class_split = {'train': {'features': scaled_train, 'classes': train_classes, 'feature_cols': head_cols}, 'validate': {'features': scaled_valid, 'classes': valid_classes, 'feature_cols': head_cols}, 'test': {'features': scaled_test, 'classes': test_classes, 'feature_cols': head_cols}}
    
    feature_data = addPhyla({'train': train_features, 'valid': valid_features, 'test': test_features})
    train_features = feature_data['train']
    valid_features = feature_data['valid']
    test_features = feature_data['test']
    phyla_cols = train_features.columns
    
    scaler = preprocessing.StandardScaler().fit(train_features)
    scaled_train = scaler.transform(train_features)
    scaled_train = pd.DataFrame(scaled_train)
    scaled_train.columns = phyla_cols
    scaled_valid = scaler.transform(valid_features)
    scaled_valid = pd.DataFrame(scaled_valid)
    scaled_valid.columns = phyla_cols
    scaled_test = scaler.transform(test_features)
    scaled_test = pd.DataFrame(scaled_test)
    scaled_test.columns = phyla_cols
    
    feature_class_phyla = {'train': {'features': scaled_train, 'classes': train_classes, 'feature_cols': phyla_cols}, 'validate': {'features': scaled_valid, 'classes': valid_classes, 'feature_cols': phyla_cols}, 'test': {'features': scaled_test, 'classes': test_classes, 'feature_cols': phyla_cols}}
    
    
    #Poly features added here
    
    train_fit = preprocessing.PolynomialFeatures(degree=2, include_bias=False).fit(train_features)
    poly_columns = []
    poly_arr = train_fit.powers_
    for i in range(0, len(poly_arr)):
        out_name = 'F'
        for j in range(0, len(poly_arr[i])):
            if poly_arr[i][j] > 0:
                out_name = out_name + '_' + str(j) + '^' + str(poly_arr[i][j])
        poly_columns.append(out_name)
    #print (poly_columns)
    train_features = train_fit.transform(train_features)
    train_features = pd.DataFrame(train_features)
    train_features.columns = poly_columns
    valid_features = preprocessing.PolynomialFeatures(degree=2, include_bias=False).fit_transform(valid_features)
    valid_features = pd.DataFrame(valid_features)
    valid_features.columns = poly_columns
    test_features = preprocessing.PolynomialFeatures(degree=2, include_bias=False).fit_transform(test_features)
    test_features = pd.DataFrame(test_features)
    test_features.columns = poly_columns
    #Feature selection added here
    selector = feature_selection.VarianceThreshold(0.00001).fit(train_features)
    select_support = selector.get_support(True)
    select_feature_cols = [poly_columns[i] for i in select_support]
    train_features = selector.transform(train_features)
    valid_features = selector.transform(valid_features)
    test_features = selector.transform(test_features)
    
    scaler = preprocessing.StandardScaler().fit(train_features)
    scaled_train = scaler.transform(train_features)
    scaled_train = pd.DataFrame(scaled_train)
    scaled_train.columns = select_feature_cols
    scaled_valid = scaler.transform(valid_features)
    scaled_valid = pd.DataFrame(scaled_valid)
    scaled_valid.columns = select_feature_cols
    scaled_test = scaler.transform(test_features)
    scaled_test = pd.DataFrame(scaled_test)
    scaled_test.columns = select_feature_cols
    
    feature_class_select = {'train': {'features': scaled_train, 'classes': train_classes, 'feature_cols': head_cols}, 'validate': {'features': scaled_valid, 'classes': valid_classes, 'feature_cols': head_cols}, 'test': {'features': scaled_test, 'classes': test_classes, 'feature_cols': head_cols}}    
    
    return_features = [feature_class_split]
    if with_phyla:
        return_features.append(feature_class_phyla)
    if with_poly:
        return_features.append(feature_class_select)
    return return_features
    