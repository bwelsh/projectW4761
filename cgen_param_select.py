import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn import cross_validation as crossv
from sklearn import linear_model, metrics, decomposition, tree, naive_bayes, ensemble, neighbors, svm, grid_search, feature_selection, preprocessing
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
import random, csv, math
from cgen_include import *
from cgen_pre_perio import *
from cgen_naive import *
from cgen_pre_richness import *
from cgen_catalog import *

### Functions ###

def bestModel(parameters, model, train_features, train_classes, valid_features, valid_classes, ml):
    '''
    Given a model with parameters and data, this function uses gridsearch to find the best parameters for the model, then assesses the fit of the model and returns a dictionary with the best parameters, score, model name and roc (sensitivity+specificity) score
    '''
    clf = grid_search.GridSearchCV(model, parameters)
    clf = clf.fit(train_features, np.ravel(train_classes))
    result = clf.predict(valid_features)
    best_params = clf.best_params_
    confusion = assessFit(result, valid_classes)
    #The JSON parser doesn't like these base estimators when writing to file. Delete them and just save the name of the base estimator used
    base = None
    if ml in ['AdaBoost', 'Bagging']:
        base = str(best_params['base_estimator'])[:1]
        del best_params['base_estimator']
    return {'params': best_params, 'score': clf.best_score_, 'classifier': ml, 'roc': confusion['TPR']+(1-confusion['FPR']), 'base': base}
    
def selectBestParameters(parameters, feature_class_split):
    '''
    Given a dictonary with models and parameter ranges and another with data, loop through all models and find the best parameters for each model with the given data
    '''
    best_params = {}
    for ml in parameters.keys():
        best_params[ml] = bestModel(parameters[ml]['features'], parameters[ml]['model'], feature_class_split['train']['features'], feature_class_split['train']['classes'], feature_class_split['valid']['features'], feature_class_split['valid']['classes'], ml)
    return best_params
    
def findMaxDepth(num_features):
    '''
    Given a number of features, this creates a range to use in decision tree family models for the max depth parameter
    '''
    median = int(math.ceil(num_features/2))
    depth = list(range(max(median-6, 1), min(median+6,num_features), 2))
    return depth
    
def aggregateBestParameters(datasets, parameters, random_seed, num_levels, file_ix):
    '''
    Given a dictionary of datasets with available taxa levels, a list of models with parameter ranges, a random_seed value (for consistency), a number of taxa levels to include and a file index, for each datasets in the dataset dictionary, create a list of all combinations of taxa levels for num_levels passed in, get the data for that dataset for each combination of taxat levels found, then find the best parameters for each model for that data and the parameter ranges passed in for that model, and put all that information into a dictionary by dataset/taxa level key. Finally, write this dictionary to a file to be read back in for analysis.
    '''
    all_best_params = {}
    for ds in datasets:
        level_combos = []
        for i in num_levels:
            level_combos.extend(combinations(datasets[ds], i))
        for levels in level_combos:
            #TODO something like this is used in cgen_project as well - might be better to make a function out of this
            if ds == 'perio':
                data = getPerioData(random_seed, levels)
                curr_data = splitDataset(data, random_seed)
                curr_data = scaleFeatures(curr_data)
            elif ds == 'richness':
                data = getRichnessData(random_seed, levels, 30)
                curr_data = splitDataset(data, random_seed)
                curr_data = scaleFeatures(curr_data)
            elif ds == 'catalog':
                data = getCatalogData(random_seed, levels, 30)
                curr_data = splitDataset(data, random_seed)
                curr_data = scaleFeatures(curr_data)
            elif ds == 'naive':
                curr_data = getOrigSplitData(random_seed, levels)
                curr_data = scaleFeatures(curr_data)
            else:
                p_data = scaleFeatures(splitDataset(getPerioData(random_seed, levels), random_seed))
                r_data = scaleFeatures(splitDataset(getRichnessData(random_seed, levels, 30), random_seed))
                c_data = scaleFeatures(splitDataset(getCatalogData(random_seed, levels, 30), random_seed))
                n_data = scaleFeatures(getOrigSplitData(random_seed, levels))
                curr_data = {}
                for d_type in ['train', 'valid', 'test']:
                    curr_data[d_type] = {}
                    for r_type in ['features', 'classes']:
                        curr_data[d_type][r_type] = combineDatasets([p_data[d_type][r_type], r_data[d_type][r_type], c_data[d_type][r_type], n_data[d_type][r_type]])
                    curr_data[d_type]['feature_cols'] = list(curr_data[d_type]['features'].columns.values)
            if 'Forest' in parameters:
                parameters['Forest']['features']['max_depth'] = findMaxDepth(len(curr_data['train']['features'].columns))
            selected = selectBestParameters(parameters, curr_data)
            all_best_params[ds+'.'+';'.join(levels)] = selected
    writeFittedData(all_best_params, 'fit'+str(file_ix)+'.json')
    
def writeFittedData(data, out_file_name):
    '''
    Given a dictionary and a file name, write this dictionary to the filename provided in json format.
    '''
    with open(out_file_name, 'w') as out_file:
        json.dump(data, out_file)

    
### Main ###

random_seed = 4

#Models with parameter ranges to test
forest = {'features': {'n_estimators': [20], 'max_features': [None, 'sqrt', 0.2, 0.3, 0.4], 'min_samples_leaf': [1, 3, 5, 7, 9], 'class_weight': ['auto'], 'random_state': [random_seed]}, 'model': ensemble.RandomForestClassifier()}

adaboost = {'features': {'base_estimator': [tree.DecisionTreeClassifier(max_depth=8, random_state=random_seed), ensemble.AdaBoostClassifier(n_estimators = 20, random_state=random_seed), ensemble.RandomForestClassifier(max_depth=5, random_state=random_seed)], 'n_estimators': [20], 'learning_rate': [1, 0.1, 0.001, 0.0001], 'random_state': [random_seed]}, 'model': ensemble.AdaBoostClassifier()}

bagging = {'features': {'base_estimator': [tree.DecisionTreeClassifier(max_depth=8, random_state=random_seed)], 'max_samples': [1, 0.75, 0.5, 0.25], 'max_features': [1, 0.75, 0.5], 'n_estimators': [20], 'bootstrap': [True, False], 'bootstrap_features': [True, False], 'random_state': [random_seed]}, 'model': ensemble.BaggingClassifier()}

logit = {'features': {'C': [0.001, 0.01, 0.1, 1, 100, 1000, 10000], 'solver': ['newton-cg', 'lbfgs', 'liblinear'], 'class_weight': ['auto'], 'random_state': [random_seed]}, 'model': linear_model.LogisticRegression()}

knn = {'features': {'n_neighbors': list(range(5, 25, 5)), 'weights': ['uniform', 'distance']},'model': neighbors.KNeighborsClassifier()}

c_svm = {'features': {'C': [0.1, 1, 100], 'gamma': [0.001, 0.1, 10], 'kernel': ['rbf'], 'class_weight': ['auto'], 'random_state': [random_seed]}, 'model': svm.SVC()}

perceptron = {'features': {'penalty': [None, 'l2', 'l1', 'elasticnet'], 'random_state': [random_seed]}, 'model': linear_model.Perceptron()}

n_bayes = {'features': {}, 'model': naive_bayes.GaussianNB()}

pca_logit = {'features': {'pca__n_components': [0.001, 0.33, 0.67], 'logit__C': [0.001, 0.01, 1, 100, 1000], 'logit__random_state': [random_seed]}, 'model': Pipeline(steps=[('pca', decomposition.PCA()), ('logit', linear_model.LogisticRegression())])}

neural = {'features': {'rbm__learning_rate': [0.001, 0.01, 0.1, 1], 'rbm__n_iter': [20], 'rbm__n_components': [100], 'rbm__random_state': [random_seed], 'logit__C': [0.001, 0.01, 1, 100, 1000], 'logit__random_state': [random_seed]}, 'model': Pipeline(steps=[('rbm', BernoulliRBM()), ('logit', linear_model.LogisticRegression())])}

passive_aggressive = {'features': {'C': [0.001, 0.01, 0.1], 'loss': ['hinge', 'squared_hinge'], 'random_state': [random_seed]}, 'model': linear_model.PassiveAggressiveClassifier()}

#Create the parameter/model dictionaries
parameters = {'Forest': forest, 'Logit': logit, 'KNN': knn, 'Neural': neural, 'SVM':  c_svm, 'Perceptron': perceptron, 'NaiveBayes': n_bayes, 'PCALogit': pca_logit, 'PassAggress': passive_aggressive, 'AdaBoost': adaboost, 'Bagging': bagging}

#Dictionary of datasets along with the available taxa levels for each one
datasets = {'richness': ['genus', 'family', 'order', 'class', 'phylum', 'domain'], 'perio': ['domain', 'phylum', 'class', 'order', 'family', 'genus', 'species'], 'catalog': ['domain', 'phylum', 'class', 'order', 'family', 'genus', 'species'], 'naive': ['order', 'phylum'], 'multi': ['order', 'phylum']}

p_dataset = {'perio': ['domain', 'phylum', 'class', 'order', 'family', 'genus', 'species']}

r_dataset = {'richness': ['genus', 'family', 'order', 'class', 'phylum', 'domain']}

c_dataset = {'catalog': ['domain', 'phylum', 'class', 'order', 'family', 'genus', 'species']}

n_dataset = {'naive': ['order', 'phylum']}

m_dataset = {'multi': ['order', 'phylum']}

#Find the best parameters for each dataset/taxa levels/model combination and save the results to files
aggregateBestParameters(datasets, parameters, random_seed, [2], 1) 
aggregateBestParameters(datasets, parameters, random_seed, [1], 2) 
aggregateBestParameters(p_dataset, parameters, random_seed, [3], 3) 
aggregateBestParameters(r_dataset, parameters, random_seed, [3], 4) 
aggregateBestParameters(c_dataset, parameters, random_seed, [3], 5) 
