import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cross_validation as crossv
from sklearn import linear_model, metrics, decomposition, tree, naive_bayes, ensemble, neighbors, cluster, svm, grid_search, preprocessing, feature_selection
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
import random
import csv
import math
import sklearn
from cgen_include import *
from cgen_pre_perio import *
from cgen_naive import *
from cgen_pre_richness import *
from cgen_catalog import *
from itertools import combinations

###Functions###
    
def findBestModel(ml_type, parameters, model, train_features, train_classes, valid_features, valid_classes):
    '''
    Given a model with parameters and data, this function fits the model, then predicts based on this fit and assesses the fit of the model, returning the True Positive Rate and False Positive Rate
    '''
    #TODO I really should be getting the random state from a passed in value here
    #TODO The base parameters will not work right for AdaBoost if the best base is not a tree, this needs to be fixed
    bases = {'Bagging': {'D': tree.DecisionTreeClassifier(max_depth=8, random_state=4)}, 'AdaBoost': {'D': tree.DecisionTreeClassifier(max_depth=8, random_state=4), 'A': ensemble.AdaBoostClassifier(n_estimators = 20, random_state=4), 'R': ensemble.RandomForestClassifier(max_depth=5, random_state=4)}}
    x = []
    y = []
    clf = model
    clf.set_params(**parameters)
    if ml_type in bases.keys():
        clf.set_params(base_estimator=bases[ml_type]['D'])
    clf = clf.fit(train_features, np.ravel(train_classes))
    result = clf.predict(valid_features)
    n_result = [x if x==1 else -1 for x in result]
    result = n_result
    ft = assessFit(result, valid_classes)
    y.append(ft['TPR'])
    x.append(ft['FPR'])
    return x, y
  
def plotTechniques(data, parameters):
    '''
    Given a dictionary of data from the parameter selection and parameters, this function finds the average roc score for each model for each dataset (can change this to a min/max/median metric as noted below) and plots this data in a bar chart
    '''
    ml_order = list(parameters.keys())
    plot_data = {}
    for ds in data:
        ds_type, levels = ds.split('.')
        if ds_type not in plot_data:
            plot_data[ds_type] = {}
            for ml in parameters:
                plot_data[ds_type][ml] = []
        for tech in data[ds]:
            plot_data[ds_type][tech].append(data[ds][tech]['roc'])
    metric = {}
    labels = []
    for tech in ml_order:
        labels.append(tech[:3])
    for ds in plot_data:
        metric[ds] = []
        for tech in ml_order:
            #Change metric here to produce different graph (min, max, mean, median, etc.)
            metric[ds].append(np.mean(plot_data[ds][tech]))
    index = np.arange(len(ml_order))
    bar_width = 0.15
    fig, ax = plt.subplots()
    colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854']
    count = 0
    for ds in metric:
        plt.bar(index+count*bar_width, metric[ds], bar_width, color=colors[count], label=ds)
        count = count+1  
                 
    plt.xlabel('Technique')
    plt.ylabel('Sensitivity + Specificity')
    plt.title('Mean Score by Technique')
    plt.xticks(index+bar_width*2.5, labels)
    plt.legend(loc=4)
    plt.savefig('mean_score_technique.png', bbox_inches='tight')
    
def plotLevels(data):
    '''
    Given a dictionary of data from the parameter selection, this function finds the median roc score for each taxa level for each dataset (can change this to a min/max/median metric as noted below) and plots this data in a bar chart
    '''
    plot_data = {}
    for ds in data:
        ds_type, levels = ds.split('.')
        if ds_type not in plot_data:
            plot_data[ds_type] = {}
        if levels not in plot_data[ds_type]:
            plot_data[ds_type][levels] = []

        for tech in data[ds]:
            plot_data[ds_type][levels].append(data[ds][tech]['roc'])
    metric = {}
    interest_levels = ['domain', 'phylum', 'class', 'order', 'family', 'genus', 'species']
    for ds in plot_data:
        metric[ds] = []
        for lev in interest_levels:
            if lev in plot_data[ds]:
                #Change metric here to produce different graph (min, max, mean, median, etc.)
                metric[ds].append(np.median(plot_data[ds][lev]))
            else:
                metric[ds].append(0)
    index = np.arange(len(interest_levels))
    bar_width = 0.15
    fig, ax = plt.subplots()
    colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854']
    count = 0
    for ds in metric:
        plt.bar(index+count*bar_width, metric[ds], bar_width, color=colors[count], label=ds)
        count = count+1  
                 
    plt.xlabel('Taxa Level')
    plt.ylabel('Sensitivity + Specificity')
    plt.title('Median Score by Taxa Level')
    plt.xticks(index+bar_width*2.5, interest_levels)
    plt.legend(loc=4)
    plt.savefig('median_score_taxa.png', bbox_inches='tight')
  
def plotByDS(parameters, fig_num, data, best_params, ds, predict_sets):
    '''
    Given a dictionary of parameters, data from a dataset, the best parameters as calculated previously and which set(s) to use for assessing fit (validation set, test set or both), this function goes through each model and fits the model with the best parameters as previously calculated, getting the TPR and FPR for the prediction set, then plotting those on a scatterplot by model.
    '''
    fig = plt.figure(fig_num)
    for ml in parameters:
        for p_type in predict_sets:
            x, y = findBestModel(ml, best_params[ml]['params'], parameters[ml]['model'], data['train']['features'], data['train']['classes'], data[p_type]['features'], data[p_type]['classes'])
            if p_type == 'valid':
                plt.scatter(x, y, c=parameters[ml]['color'], label = ml, linewidth=0, s=50, marker=predict_sets[p_type])
            else:
                plt.scatter(x, y, c=parameters[ml]['color'], linewidth=0, s=50, marker=predict_sets[p_type])
        plt.legend(loc=4)
    plt.plot([0, 0.25, 0.5, 0.75, 1], [0, 0.25, 0.5, 0.75, 1], c='black')
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.xlabel('False Positive')
    plt.ylabel('True Positive')
    plt.title('ROC Space for different ML Techniques: ' + ds + ' dataset')
    plt.savefig('best_level_' + ds + '.png', bbox_inches='tight')
    
def plotBest(parameters, fig_num, data, best_params, datasets, predict_sets):
    '''
    Given a dictionary of parameters, data from all datasets, the best parameters as calculated previously and which set(s) to use for assessing fit (validation set, test set or both), this function goes through each dataset and fits the best model only for that dataset with the best parameters as previously calculated, getting the TPR and FPR for the prediction set, then plotting those on a scatterplot by dataset.
    '''
    fig = plt.figure(fig_num)
    colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854']
    count = 0
    for ds in datasets:
        ml = datasets[ds]
        for p_type in predict_sets:
            x, y = findBestModel(ml, best_params[ds][ml]['params'], parameters[ml]['model'], data[ds]['train']['features'], data[ds]['train']['classes'], data[ds][p_type]['features'], data[ds][p_type]['classes'])
            if p_type == 'valid':
                plt.scatter(x, y, c=colors[count], label = ds+': '+ml, linewidth=0, s=50, marker=predict_sets[p_type])
            else:
                plt.scatter(x, y, c=colors[count], linewidth=0, s=50, marker=predict_sets[p_type])
        count = count + 1
        plt.legend(loc=4)
    plt.plot([0, 0.25, 0.5, 0.75, 1], [0, 0.25, 0.5, 0.75, 1], c='black')
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.xlabel('False Positive')
    plt.ylabel('True Positive')
    plt.title('ROC Space for Best Technique By Dataset')
    plt.savefig('best_level_all.png', bbox_inches='tight')
    
def getBestLevels(data, num_best):
    '''
    Given data on model fit as previously calculated and a parameter for the top number of levels to return, this function goes through each dataset and orders the taxa levels by best to worst roc score, then chooses the top number of them (according to the value passed in) and puts them in a dictionary to return
    '''
    best_levels = {}
    best_score = {}
    for ds in data:
        ds_type, levels = ds.split('.')
        if ds_type not in best_score:
            best_score[ds_type] = {'max_score': 0, 'max_levels': []}
        curr_total = []
        for tech in data[ds]:
            curr_total.append(data[ds][tech]['roc'])
        curr_total = sorted(curr_total, reverse=True)[:num_best]
        curr_avg = np.mean(curr_total)
        if curr_avg > best_score[ds_type]['max_score']:
            best_score[ds_type]['max_score'] = curr_avg
            best_score[ds_type]['max_levels'] = [levels]
        elif curr_avg == best_score[ds_type]['max_score']:
            best_score[ds_type]['max_levels'].append(levels)
        best_levels[ds_type] = best_score[ds_type]['max_levels']
    return best_levels
    
def getBestFitData(best_fit, random_seed, with_poly, with_fs):
    '''
    Given data on model fit as previously calculated and a flag for poly features and feature selection, this function goes through each dataset and gets the data for the best fit model, scaling and splitting it for future analysis. If either the poly or feature selection flags are set to true, this function also will add polynomial features or perform feature selection before returning the feature and class dataframes for the train/valid/test sets for each dataset
    '''
    best = {}
    for ds in best_fit:
        if ';' in best_fit[ds][0]:
            levels = best_fit[ds][0].split(';')
        else:
            levels = [best_fit[ds][0]]
        if ds == 'multi':
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
        elif ds == 'naive':
            curr_data = getOrigSplitData(random_seed, levels)
        else:
            if ds == 'perio':
                data = getPerioData(random_seed, levels)
            elif ds == 'richness':
                data = getRichnessData(random_seed, levels, 30)
            else:
                data = getCatalogData(random_seed, levels, 30)
            curr_data = splitDataset(data, random_seed)
        if with_poly:
            curr_data = addPolyFeatures(curr_data, 2) 
        if with_fs:
            curr_data = selectFeatures(curr_data, 100)
        curr_data = scaleFeatures(curr_data)
        best[ds] = curr_data
    return best
    
def getBestTechnique(techs):
    '''
    Given data on model fit as previously calculated, this function finds the model with the best score and returns it
    '''
    max_score = 0
    max_tech = ''
    for ml in techs:
        if techs[ml]['roc'] > max_score:
            max_score = techs[ml]['roc']
            max_tech = ml
    return max_tech
    
def getBestForest(data):
    '''
    Given data on model fit as previously calculated, this function finds the best random forest for each dataset, returning the taxa levels that need to be used to get this best scoring forest
    '''
    best_forest = {}
    for ds in data:
        ds_type, levels = ds.split('.')
        if ds_type not in best_forest:
            best_forest[ds_type] = {'max_score': 0, 'max_levels': []}
        curr_forest = data[ds]['Forest']['roc']
        if curr_forest > best_forest[ds_type]['max_score']:
            best_forest[ds_type]['max_score'] = curr_forest
            best_forest[ds_type]['max_levels'] = levels
    best = {}
    for ds in best_forest:
        best[ds] = [best_forest[ds]['max_levels']]
    return best
    
def getBestForestFeatures(parameters, random_seed, best_fit, best_forests, data):
    '''
    Given data on model fit as previously calculated, a dictionary of models, a dictionary of the levels to retreive the best forest for each dataset and the dataframes for that best level, this function fits the best forest and then gets the features importances for that forest. It then sorts by most important for each dataset and writes the results to a file.
    '''
    clf = parameters['Forest']['model']
    output = ''
    for ds in best_forests:
        curr_params = best_fit[ds+'.'+best_forests[ds][0]]['Forest']['params']
        clf.set_params(**curr_params)
        clf = clf.fit(data[ds]['train']['features'], np.ravel(data[ds]['train']['classes']))
        importances = clf.feature_importances_
        features = data[ds]['train']['features'].columns.values
        feat_import = []
        for i in range(len(features)):
            feat_import.append((features[i], importances[i]))
        feat_import = sorted(feat_import,key=lambda x: x[1], reverse=True)
        output = output + ds + '\n'
        for i in range (len(feat_import)):
            output = output + feat_import[i][0] + '\t' + str("%.3f" % float(feat_import[i][1])) + '\n'
        output = output + '\n'  
    f = open('feature_importances.txt', 'w')
    f.write(output)
    f.close()
    
def loadFittedData(file_range):
    '''
    Given a range of file indices, this function goes through those files and uses the json parser to load the data, then combines all of the data into a single dictionary and returns it.
    '''
    in_data = {}
    for i in range(file_range[0], file_range[1]+1):
        in_file_name = 'fit' + str(i) + '.json'
        with open(in_file_name) as in_file:
            part_data = json.load(in_file)
            for run_type in part_data:
                if run_type in in_data:
                    for ml_tech in part_data[run_type]:
                        in_data[run_type][ml_tech] = part_data[run_type][ml_tech]
                else:
                    in_data[run_type] = part_data[run_type]
    return in_data
            
###Main###

random_seed = 4
random.seed(random_seed)

#Models
forest = {'color': '#6a3d9a', 'model': ensemble.RandomForestClassifier()}

adaboost = {'color': '#a6cee3', 'model': ensemble.AdaBoostClassifier()}

bagging = {'color': '#1f78b4', 'model': ensemble.BaggingClassifier()}

logit = {'color': '#b2df8a', 'model': linear_model.LogisticRegression()}

knn = {'color': '#33a02c', 'model': neighbors.KNeighborsClassifier()}

c_svm = {'color': '#fb9a99', 'model': svm.SVC()}

perceptron = {'color': '#e31a1c', 'model': linear_model.Perceptron()}

n_bayes = {'color': '#fdbf6f', 'model': naive_bayes.GaussianNB()}

pca_logit = {'color': '#ff7f00', 'model': Pipeline(steps=[('pca', decomposition.PCA()), ('logit', linear_model.LogisticRegression())])}

neural = {'color': '#cab2d6', 'model': Pipeline(steps=[('rbm', BernoulliRBM()), ('logit', linear_model.LogisticRegression())])}

passive_aggressive = {'color': '#ffff99', 'model': linear_model.PassiveAggressiveClassifier()}
    
parameters = {'Forest': forest, 'AdaBoost': adaboost, 'Logit': logit, 'KNN': knn, 'Neural': neural, 'SVM':  c_svm, 'Perceptron': perceptron, 'Bagging': bagging, 'NaiveBayes': n_bayes, 'PCALogit': pca_logit, 'PassAggress': passive_aggressive}

#Load data that was the output of cgen_param_select.py
data = loadFittedData((1,5))

#Plots w/o data
#plotTechniques(data, parameters)
#plotLevels(data)

#Getting data for best levels and more plots
'''
best_levels = getBestLevels(data, 5)
best_data = getBestFitData(best_levels, random_seed, False, False)
'''
#count = 1
#for ds in best_levels:
    #plotByDS(parameters, count, best_data[ds], data[ds+'.'+best_levels[ds][0]], ds, {'valid': 'o', 'test': 'v'})
    #count = count + 1
#best_params = {}
#best_tech = {}
#for ds in best_levels:
    #best_params[ds] = data[ds+'.'+best_levels[ds][0]]
    #best_tech[ds] = getBestTechnique(best_params[ds])
#plotBest(parameters, 1, best_data, best_params, best_tech, {'valid': 'o', 'test': 'v'})

#Getting best forest and fitting to get feature importances
#best_forests = getBestForest(data)
#best_forest_data = getBestFitData(best_forests, random_seed, False, False)
#getBestForestFeatures(parameters, random_seed, data, best_forests, best_forest_data)


