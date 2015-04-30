import numpy as np
import pandas as pd
import math
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing, feature_selection
import copy

###Functions###
def extractFeatures(all_data):
    split_data = {}
    for data in all_data:
        features = all_data[data].drop(['Diagnosis', 'Sample'], axis=1)
        feature_cols = features.columns
        other = all_data[data][['Diagnosis']]
        split_data[data] = {}
        split_data[data]['features'] = features
        split_data[data]['feature_cols'] = feature_cols
        split_data[data]['class_data'] = other
    return split_data
    
def scaleData(scaler, updated_cols, feature_data):
    scaled_data = {}
    for data in feature_data:
        scaled = scaler.transform(feature_data[data])
        scaled = pd.DataFrame(scaled)
        scaled.columns = updated_cols
        scaled_data[data] = scaled
    return scaled_data
    
def combineFeaturesAndClasses(scaled_data, feature_data, headers):
    split_data = {}
    for data in scaled_data:
        split_data[data] = {}
        split_data[data]['features'] = scaled_data[data]
        split_data[data]['classes'] = feature_data[data]['class_data']
        split_data[data]['feature_cols'] = headers
    return split_data
    
def combineDatasets(datasets):
    inter_cols = datasets[0].columns.values
    new_sets = []
    for dset in datasets:
        inter_cols = list(set(inter_cols) & set(dset.columns.values))
    for dset in datasets:
        new_sets.append(dset[inter_cols])
    return (pd.concat(new_sets))
    
def splitScaleSeparate(data, random_seed, add_poly_features, deg, threshold):
    #Get column headers
    col_headers = list(data.columns.values)
    feature_cols = copy.deepcopy(col_headers)
    feature_cols.remove('Sample')
    feature_cols.remove('Diagnosis')
    class_col = ['Diagnosis']
    
    #Train/test/validate split
    train, test = train_test_split(data, test_size=0.2, random_state=random_seed)
    train = pd.DataFrame(train)
    test = pd.DataFrame(test)
    train.columns = col_headers
    test.columns = col_headers
    train, validate = train_test_split(train, test_size=0.25, random_state=random_seed)
    train = pd.DataFrame(train)
    validate = pd.DataFrame(validate)
    train.columns = col_headers
    validate.columns = col_headers
    
    #Separate features and classes
    all_data = {'train': train, 'valid': validate, 'test': test}
    feature_data = extractFeatures(all_data)
    train_features = feature_data['train']['features']
    valid_features = feature_data['valid']['features']
    test_features = feature_data['test']['features']
    
    #Add poly features if requested and select
    if add_poly_features:
        train_fit = preprocessing.PolynomialFeatures(degree=deg, include_bias=False).fit(train_features)
        new_columns = []
        poly_arr = train_fit.powers_
        for i in range(0, len(poly_arr)):
            out_name = 'F'
            for j in range(0, len(poly_arr[i])):
                if poly_arr[i][j] > 0:
                    out_name = out_name + '_' + str(j) + '^' + str(poly_arr[i][j])
            new_columns.append(out_name)
        train_features = train_fit.transform(train_features)
        train_features = pd.DataFrame(train_features)
        train_features.columns = new_columns
        valid_features = preprocessing.PolynomialFeatures(degree=deg, include_bias=False).fit_transform(valid_features)
        valid_features = pd.DataFrame(valid_features)
        valid_features.columns = new_columns
        test_features = preprocessing.PolynomialFeatures(degree=deg, include_bias=False).fit_transform(test_features)
        test_features = pd.DataFrame(test_features)
        test_features.columns = new_columns
        selector = feature_selection.VarianceThreshold(threshold).fit(train_features)
        select_support = selector.get_support(True)
        updated_cols = [new_columns[i] for i in select_support]
        train_features = selector.transform(train_features)
        valid_features = selector.transform(valid_features)
        test_features = selector.transform(test_features)
    else:
        updated_cols = feature_cols
    
    #Scale data to same scale, using train data to find the scale
    scaler = preprocessing.MinMaxScaler().fit(train_features)
    to_scale_data = {'train': train_features, 'valid': valid_features, 'test': test_features}
    scaled_data = scaleData(scaler, updated_cols, to_scale_data)
    return combineFeaturesAndClasses(scaled_data, feature_data, feature_cols)
