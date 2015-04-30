import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn import cross_validation as crossv
from sklearn import linear_model, metrics, decomposition
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
from cgen_include import *
from cgen_pre_perio import *
from cgen_orig_data import *
from cgen_pre_richness import *
from cgen_catalog import *
from itertools import combinations

###Functions###
    
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
    
def findBestModel(ml_type, parameters, model, train_features, train_classes, valid_features, valid_classes):
    x = []
    y = []
    name = []
    clf = grid_search.GridSearchCV(model, parameters)
    clf = clf.fit(train_features, np.ravel(train_classes))
    print(ml_type, clf.best_params_)
    result = clf.predict(valid_features)
    n_result = [x if x==1 else -1 for x in result]
    result = n_result
    print(ml_type + ":\n%s\n" % (metrics.classification_report(valid_classes.as_matrix(),result)))
    ft = assessFit(result, valid_classes)
    name.append(ml_type)
    y.append(ft['TPR'])
    x.append(ft['FPR'])
    return x, y, name
    
def fitAndPlot(parameters, feature_class_split, fig_num):
    fig = plt.figure(fig_num)
    for ml in parameters.keys():
        x = []
        y = []
        for features in feature_class_split:
            x_sub, y_sub, name_sub = findBestModel(ml, parameters[ml]['features'], parameters[ml]['model'], features['train']['features'], features['train']['classes'], features['valid']['features'], features['valid']['classes'])
            x.append(x_sub)
            y.append(y_sub)
        plt.scatter(x, y, c=parameters[ml]['color'], label = ml, linewidth=0, s=35)
        plt.legend(loc=4)
    plt.plot([0, 0.25, 0.5, 0.75, 1], [0, 0.25, 0.5, 0.75, 1], c='black')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('False Positive')
    plt.ylabel('True Positive')
    plt.title('ROC Space for different ML Techniques')
    
    plt.show()
    
def bestModel(parameters, model, train_features, train_classes):
    clf = grid_search.GridSearchCV(model, parameters)
    clf = clf.fit(train_features, np.ravel(train_classes))
    return {'params': clf.best_params_, 'score': clf.best_score_, 'classifier': model}
    
def selectBestParameters(parameters, feature_class_split):
    best_params = {}
    for ml in parameters.keys():
        best_params[ml] = bestModel(parameters[ml]['features'], parameters[ml]['model'], feature_class_split['train']['features'], feature_class_split['train']['classes'])
    return best_params
    
def aggregateBestParameters(datasets, parameters, random_seed):
    #TODO fix calling of correct function
    #TODO fix range
    all_best_params = {}
    for ds in datasets:
        level_combos = []
        for i in range(1):
            level_combos.extend(combinations(datasets[ds]['taxas'], i+1))
        for levels in level_combos:
            data = getPerioData(random_seed, levels)
            for feature_types in ['poly1']:
                curr_data = splitScaleSeparate(data, random_seed, False, 2, 0)
                selected = selectBestParameters(parameters, curr_data)
                all_best_params[ds+'.'+feature_types+'.'+';'.join(levels)] = selected
    return all_best_params
            
###Main###
random_seed = 4
random.seed(random_seed)

### ML starts here

#TODO fix the way the ranges are determined (try log range in np)
#TODO implement Neural networks with this new model
#TODO try other kernels for SVM
#TODO additional perceptron models 
# forest features list(range(3, 8)), list(range(2, 11))
#TODO Fix max_depth to be percentage of samples
#TODO Fix parameters to be exponential
#TODO Adaboost with other base_estimators
single_tree = {'features': {'criterion': ['gini', 'entropy'], 'max_depth': [2, 3, 4, 5, 6, 7], 'max_features': [None, 'sqrt', 0.2, 0.3, 0.4], 'min_samples_leaf': [1, 3, 5, 7, 9], 'class_weight': ['auto'], 'random_state': [random_seed]}, 'color': '#4daf4a', 'model': tree.DecisionTreeClassifier()}

forest = {'features': {'n_estimators': [20], 'max_depth': [5, 6, 7], 'max_features': ['sqrt', 0.2], 'min_samples_leaf': [1, 3], 'class_weight': ['auto'], 'random_state': [random_seed]}, 'color': '#984ea3', 'model': ensemble.RandomForestClassifier()}

#forest = {'features': {'n_estimators': [20], 'max_depth': [2, 3, 4, 5, 6, 7], 'max_features': [None, 'sqrt', 0.2, 0.3, 0.4], 'min_samples_leaf': [1, 3, 5, 7, 9], 'class_weight': ['auto'], 'random_state': [random_seed]}, 'color': '#984ea3', 'model': ensemble.RandomForestClassifier()}

extra_trees = {'features': {'n_estimators': [100], 'max_depth': [2, 3, 4, 5, 6, 7], 'max_features': [None, 'sqrt', 0.2, 0.3, 0.4], 'min_samples_leaf': [1, 3, 5, 7, 9], 'class_weight': ['auto'], 'random_state': [random_seed]}, 'color': 'purple', 'model': ensemble.ExtraTreesClassifier()}

adaboost = {'features': {'base_estimator': [tree.DecisionTreeClassifier(max_depth=8, random_state=random_seed)], 'n_estimators': [20], 'learning_rate': [1, 0.1, 0.001, 0.0001], 'random_state': [random_seed]}, 'color': '#ff7f00', 'model': ensemble.AdaBoostClassifier()}
#tree.ExtraTreeClassifier(max_depth=8, random_state=random_seed), linear_model.Perceptron(random_state=random_seed)

gradient_boosting = {'features': {'max_depth': [2, 3, 4, 5, 6, 7], 'max_features': [None, 'sqrt', 0.2, 0.3, 0.4], 'min_samples_leaf': [1, 3, 5, 7, 9], 'n_estimators': [100], 'learning_rate': [1, 0.1, 0.001, 0.0001], 'random_state': [random_seed]}, 'color': 'pink', 'model': ensemble.GradientBoostingClassifier()}

bagging = {'features': {'base_estimator': [tree.DecisionTreeClassifier(random_state=random_seed)], 'max_samples': [1, 0.75, 0.5, 0.25], 'max_features': [1, 0.75, 0.5, 0.25], 'n_estimators': [20], 'bootstrap': [True, False], 'bootstrap_features': [True, False], 'class_weight': ['auto'], 'random_state': [random_seed]}, 'color': 'orange', 'model': ensemble.BaggingClassifier()}

logit = {'features': {'C': [0.001, 0.01, 0.1, 1, 100, 1000, 10000], 'solver': ['newton-cg', 'lbfgs', 'liblinear'], 'class_weight': ['auto'], 'random_state': [random_seed]}, 'color': '#e41a1c', 'model': linear_model.LogisticRegression()}

knn = {'features': {'n_neighbors': list(range(5, 25, 5)), 'weights': ['uniform', 'distance']}, 'color': '#377eb8', 'model': neighbors.KNeighborsClassifier()}

c_svm = {'features': {'C': [0.1, 1, 100], 'gamma': [0.001, 0.1, 10], 'kernel': ['rbf'], 'class_weight': ['auto'], 'random_state': [random_seed]}, 'color': '#e6ab02', 'model': svm.SVC()}

nu_svm = {'features': {'nu': [0.1, 0.25, 0.5, 0.75, 0.9], 'gamma': [0.001, 0.1, 10], 'kernel': ['rbf'], 'random_state': [random_seed]}, 'color': 'pink', 'model': svm.NuSVC()}

perceptron = {'features': {'penalty': [None, 'l2', 'l1', 'elasticnet'], 'random_state': [random_seed]}, 'color': 'green', 'model': linear_model.Perceptron()}

sgd = {'features': {'penalty': [None, 'l2', 'l1', 'elasticnet'], 'loss': ['hinge', 'log', 'modified_huber'], 'alpha': [0.0001, 0.001, 0.01, 0.1, 1], 'random_state': [random_seed]}, 'color': 'purple', 'model': linear_model.SGDClassifier()}

n_bayes = {'features': {}, 'color': 'blue', 'model': naive_bayes.GaussianNB()}

pca_logit = {'features': {'pca__n_components': [40], 'logit__C': [0.001, 0.01, 1, 100, 1000], 'logit__random_state': [random_seed]}, 'color': 'pink', 'model': Pipeline(steps=[('pca', decomposition.PCA()), ('logit', linear_model.LogisticRegression())])}

neural = {'features': {'rbm__learning_rate': [0.001, 0.01, 0.1, 1], 'rbm__n_iter': [20], 'rbm__n_components': [100], 'rbm__random_state': [random_seed], 'logit__C': [0.001, 0.01, 1, 100, 1000], 'logit__random_state': [random_seed]}, 'color': 'purple', 'model': Pipeline(steps=[('rbm', BernoulliRBM()), ('logit', linear_model.LogisticRegression())])}

passive_aggressive = {'features': {'C': [0.001, 0.01, 1, 100, 1000], 'loss': ['hinge', 'squared_hinge'], 'random_state': [random_seed]}, 'color': 'red', 'model': linear_model.PassiveAggressiveClassifier()}
    
#parameters = {'Forest': forest, 'AdaBoost': adaboost, 'Logit': logit, 'KNN': knn, 'Neural': neural, 'SVM':  c_svm, 'Perceptron': perceptron, 'Bagging': bagging, 'NaiveBayes': n_bayes, 'PCALogit': pca_logit, 'PassAggress': passive_aggressive}

parameters = {'Logit': logit, 'KNN': knn, 'PCALogit': pca_logit, 'Neural': neural}

datasets = {'perio': {'taxas': ['domain', 'phylum', 'class', 'order', 'family', 'genus', 'species']}}

#res = aggregateBestParameters(datasets, parameters, random_seed)   
#print (res)

#TODO pipeline functions
#TODO scoring
#TODO can I extract other features?
#TODO additional ML techniques (mess with existing and any others)
#TODO multi-class classification

#curr_data = getOrigData(random_seed, False, True)
'''
perio_data = getPerioData(random_seed, ['phylum', 'genus'])
richness_data = getRichnessData(random_seed, ['phylum', 'genus'], 30)
catalog_data = getCatalogData(random_seed, ['phylum', 'genus'], 30)
data = combineDatasets([perio_data, richness_data, catalog_data])
'''
#curr_data = [getRichnessData(random_seed, ['phylum', 'domain'], 30, False, 2, 0.0001)]
data = getPerioData(random_seed, ['species'])
#data = getRichnessData(random_seed, ['phylum', 'genus'], 30)
#curr_data = [getPerioData(random_seed, ['phylum', 'domain'], False, 2, 0.0001), getPerioData(random_seed, ['phylum', 'domain'], True, 2, 0.0001)]

curr_data = [splitScaleSeparate(data, random_seed, False, 2, 0)]
fitAndPlot(parameters, curr_data, 1)
