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
        split_data[data]['classes'] = other
    return split_data
    
def combineDatasets(datasets):
    inter_cols = datasets[0].columns.values
    new_sets = []
    for dset in datasets:
        inter_cols = list(set(inter_cols) & set(dset.columns.values))
    for dset in datasets:
        new_sets.append(dset[inter_cols])
    return (pd.concat(new_sets))
    
def splitDataset(data, random_seed):
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
    return extractFeatures(all_data)
    
def addPolyFeatures(data, deg):
    train_features = data['train']['features']
    valid_features = data['valid']['features']
    test_features = data['test']['features']
    
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
    data['train']['features'] = train_features
    valid_features = preprocessing.PolynomialFeatures(degree=deg, include_bias=False).fit_transform(valid_features)
    valid_features = pd.DataFrame(valid_features)
    valid_features.columns = new_columns
    data['valid']['features'] = valid_features
    test_features = preprocessing.PolynomialFeatures(degree=deg, include_bias=False).fit_transform(test_features)
    test_features = pd.DataFrame(test_features)
    test_features.columns = new_columns
    data['test']['features'] = test_features
    return data
    
def selectFeatures(data, num_best):
    num_best = min(num_best, len(data['train']['features'].columns.values))
    selector = feature_selection.SelectKBest(k=num_best).fit(data['train']['features'], data['train']['classes'])
    select_support = selector.get_support(True)
    updated_cols = [data['train']['features'].columns[i] for i in select_support]
    data['train']['features'] = selector.transform(data['train']['features'])
    data['valid']['features'] = selector.transform(data['valid']['features'])
    data['test']['features'] = selector.transform(data['test']['features'])
    data['train']['features'] = pd.DataFrame(data['train']['features'])
    data['valid']['features'] = pd.DataFrame(data['valid']['features'])
    data['test']['features'] = pd.DataFrame(data['test']['features'])
    data['train']['features'].columns = updated_cols
    data['valid']['features'].columns = updated_cols
    data['test']['features'].columns = updated_cols
    return data
    
def scaleFeatures(data):
    scaler = preprocessing.MinMaxScaler().fit(data['train']['features'])
    updated_cols = data['train']['features'].columns
    for dtype in data:
        data[dtype]['features'] = scaler.transform(data[dtype]['features'])
        data[dtype]['features'] = pd.DataFrame(data[dtype]['features'])
        data[dtype]['features'].columns = updated_cols
    return data
    
def assessFit(predict_results, actual_results):
    counts = actual_results['Diagnosis'].value_counts()
    cd_count = counts[1]
    no_count = counts[-1]
    results = {'correct': {1: 0, -1: 0}, 'incorrect': {1: 0, -1: 0}}
    for i in range(0, len(predict_results)):
        if actual_results.iloc[i,0] == 1:
            if predict_results[i] == 1:
                results['correct'][predict_results[i]] += 1
            else:
                results['incorrect'][predict_results[i]] += 1
        else:
            if predict_results[i] == -1:
                results['correct'][predict_results[i]] += 1
            else:
                results['incorrect'][predict_results[i]] += 1
    confusion = {'TPR': float(results['correct'][1])/float(results['correct'][1]+results['incorrect'][-1]), 'FPR': float(results['incorrect'][1])/float(results['incorrect'][1]+results['correct'][-1]), 'TNR': float(results['correct'][-1])/float(results['incorrect'][1]+results['correct'][-1]), 'FNR': float(results['incorrect'][-1])/float(results['correct'][1]+results['incorrect'][-1])}
    return confusion
