import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from itertools import combinations
from sklearn import cross_validation as crossv
from sklearn import linear_model, metrics, decomposition, tree, naive_bayes, ensemble, neighbors, svm, grid_search, feature_selection, preprocessing
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
import random, csv, math
from cgen_include import *
from cgen_pre_perio import *
from cgen_orig_data import *
from cgen_pre_richness import *
from cgen_catalog import *
from datetime import datetime

### Functions ###

def bestModel(parameters, model, train_features, train_classes, valid_features, valid_classes, ml):
    clf = grid_search.GridSearchCV(model, parameters)
    clf = clf.fit(train_features, np.ravel(train_classes))
    result = clf.predict(valid_features)
    best_params = clf.best_params_
    confusion = assessFit(result, valid_classes)
    base = None
    if ml in ['AdaBoost', 'Bagging']:
        base = str(best_params['base_estimator'])[:1]
        del best_params['base_estimator']
    return {'params': best_params, 'score': clf.best_score_, 'classifier': ml, 'roc': confusion['TPR']+(1-confusion['FPR']), 'base': base}
    
def selectBestParameters(parameters, feature_class_split):
    best_params = {}
    for ml in parameters.keys():
        best_params[ml] = bestModel(parameters[ml]['features'], parameters[ml]['model'], feature_class_split['train']['features'], feature_class_split['train']['classes'], feature_class_split['valid']['features'], feature_class_split['valid']['classes'], ml)
    return best_params
    
def findMaxDepth(num_features):
    median = int(math.ceil(num_features/2))
    depth = list(range(max(median-6, 1), min(median+6,num_features), 2))
    return depth
    
def aggregateBestParameters(datasets, parameters, random_seed, num_levels, file_ix):
    all_best_params = {}
    for ds in datasets:
        level_combos = []
        #max_levels-1, 
        for i in num_levels:
            level_combos.extend(combinations(datasets[ds], i))
        for levels in level_combos:
            #print (ds, levels)
            #TODO make a function to do this as it is used in cgen_project as well
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
            print (ds, levels, datetime.now().time())
    writeFittedData(all_best_params, 'fit'+str(file_ix)+'.json')
    
def writeFittedData(data, out_file_name):
    with open(out_file_name, 'w') as out_file:
        json.dump(data, out_file)

    
### Main ###

random_seed = 4

#TODO remove colors from here and put on cgen_project

#Models
forest = {'features': {'n_estimators': [20], 'max_features': [None, 'sqrt', 0.2, 0.3, 0.4], 'min_samples_leaf': [1, 3, 5, 7, 9], 'class_weight': ['auto'], 'random_state': [random_seed]}, 'color': '#984ea3', 'model': ensemble.RandomForestClassifier()}

adaboost = {'features': {'base_estimator': [tree.DecisionTreeClassifier(max_depth=8, random_state=random_seed), ensemble.AdaBoostClassifier(n_estimators = 20, random_state=random_seed), ensemble.RandomForestClassifier(max_depth=5, random_state=random_seed)], 'n_estimators': [20], 'learning_rate': [1, 0.1, 0.001, 0.0001], 'random_state': [random_seed]}, 'color': '#ff7f00', 'model': ensemble.AdaBoostClassifier()}

bagging = {'features': {'base_estimator': [tree.DecisionTreeClassifier(max_depth=8, random_state=random_seed)], 'max_samples': [1, 0.75, 0.5, 0.25], 'max_features': [1, 0.75, 0.5], 'n_estimators': [20], 'bootstrap': [True, False], 'bootstrap_features': [True, False], 'random_state': [random_seed]}, 'color': 'orange', 'model': ensemble.BaggingClassifier()}

logit = {'features': {'C': [0.001, 0.01, 0.1, 1, 100, 1000, 10000], 'solver': ['newton-cg', 'lbfgs', 'liblinear'], 'class_weight': ['auto'], 'random_state': [random_seed]}, 'color': '#e41a1c', 'model': linear_model.LogisticRegression()}

knn = {'features': {'n_neighbors': list(range(5, 25, 5)), 'weights': ['uniform', 'distance']}, 'color': '#377eb8', 'model': neighbors.KNeighborsClassifier()}

c_svm = {'features': {'C': [0.1, 1, 100], 'gamma': [0.001, 0.1, 10], 'kernel': ['rbf'], 'class_weight': ['auto'], 'random_state': [random_seed]}, 'color': '#e6ab02', 'model': svm.SVC()}

perceptron = {'features': {'penalty': [None, 'l2', 'l1', 'elasticnet'], 'random_state': [random_seed]}, 'color': 'green', 'model': linear_model.Perceptron()}

n_bayes = {'features': {}, 'color': 'blue', 'model': naive_bayes.GaussianNB()}

pca_logit = {'features': {'pca__n_components': [0.001, 0.33, 0.67], 'logit__C': [0.001, 0.01, 1, 100, 1000], 'logit__random_state': [random_seed]}, 'color': 'pink', 'model': Pipeline(steps=[('pca', decomposition.PCA()), ('logit', linear_model.LogisticRegression())])}

neural = {'features': {'rbm__learning_rate': [0.001, 0.01, 0.1, 1], 'rbm__n_iter': [20], 'rbm__n_components': [100], 'rbm__random_state': [random_seed], 'logit__C': [0.001, 0.01, 1, 100, 1000], 'logit__random_state': [random_seed]}, 'color': 'purple', 'model': Pipeline(steps=[('rbm', BernoulliRBM()), ('logit', linear_model.LogisticRegression())])}

passive_aggressive = {'features': {'C': [0.001, 0.01, 0.1], 'loss': ['hinge', 'squared_hinge'], 'random_state': [random_seed]}, 'color': 'red', 'model': linear_model.PassiveAggressiveClassifier()}

#Code
parameters = {'Forest': forest, 'Logit': logit, 'KNN': knn, 'Neural': neural, 'SVM':  c_svm, 'Perceptron': perceptron, 'NaiveBayes': n_bayes, 'PCALogit': pca_logit, 'PassAggress': passive_aggressive, 'AdaBoost': adaboost, 'Bagging': bagging}
#'PassAggress': passive_aggressive

#datasets = {'perio': ['domain', 'phylum', 'class', 'order', 'family', 'genus', 'species'], 'richness': ['genus', 'family', 'order', 'class', 'phylum', 'domain'], 'catalog': ['domain', 'phylum', 'class', 'order', 'family', 'genus', 'species']}

base_parameters = {'AdaBoost': adaboost, 'Bagging': bagging}

datasets = {'richness': ['genus', 'family', 'order', 'class', 'phylum', 'domain'], 'perio': ['domain', 'phylum', 'class', 'order', 'family', 'genus', 'species'], 'catalog': ['domain', 'phylum', 'class', 'order', 'family', 'genus', 'species'], 'naive': ['order', 'phylum'], 'multi': ['order', 'phylum']}

p_dataset = {'perio': ['domain', 'phylum', 'class', 'order', 'family', 'genus', 'species']}

r_dataset = {'richness': ['genus', 'family', 'order', 'class', 'phylum', 'domain']}

c_dataset = {'catalog': ['domain', 'phylum', 'class', 'order', 'family', 'genus', 'species']}

n_dataset = {'naive': ['order', 'phylum']}

m_dataset = {'multi': ['order', 'phylum']}

#datasets = {'naive': ['order', 'phylum'], 'multi': ['order', 'phylum']}

#datasets = {'perio': ['domain', 'phylum', 'class', 'order', 'family', 'genus', 'species']}

print (datetime.now().time())
aggregateBestParameters(datasets, parameters, random_seed, [1], 2) 
aggregateBestParameters(p_dataset, parameters, random_seed, [3], 3) 
aggregateBestParameters(r_dataset, parameters, random_seed, [3], 4) 
aggregateBestParameters(c_dataset, parameters, random_seed, [3], 5) 

#writeFittedData(best_fit, 'test_fit2.json')
#print (res['richness.poly1.phylum']['Forest'])