import numpy as np
import pandas as pd
import math
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from cgen_include import *
import copy

###Functions###

def getTaxa(file):
    taxa_order = ['domain', 'phylum', 'class', 'order', 'family', 'genus', 'species']
    tax_categories = {'domain': [], 'phylum': [], 'class': [], 'order': [], 'family': [], 'genus': [], 'species': []}
    f = open(file)
    for line in f:
        line = line.strip()
        line = line.split('\t')
        lineage = line[79]
        lineage = lineage.split(';')
        for i in range(0, len(lineage)):
            trim_lin = lineage[i]
            trim_lin = trim_lin.rsplit('_', 1)[1]
            tax_categories[taxa_order[i]].append(trim_lin)
    f.close()
    for cat in tax_categories:
        tax_categories[cat] = set(tax_categories[cat])
        tax_categories[cat] = list(tax_categories[cat])
    return tax_categories
    
def loadData(file, cats, levels):
    data = []
    headers = ['Sample', 'Diagnosis']
    for lev in cats:
        for item in cats[lev]:
            headers.append(item+"_"+lev)
        headers.append('Unknown_' + lev)
    ix = 0
    f = open(file)
    for line in f:
        if ix == 0:
            for i in range(0, 72):
                ln_data = [0] * len(headers)
                ln_data[0] = i
                if i < 33:
                    ln_data[1] = -1
                else:
                    ln_data[1] = 1
                data.append(ln_data)
        line = line.strip()
        line = line.split('\t')
        lineage = line[79]
        lineage = lineage.split(';')
        for level in levels:
            if len(lineage)-1 >= levels[level]:
                trim_lin = lineage[levels[level]]
                trim_lin = trim_lin.rsplit('_', 1)[1]
                col = headers.index(trim_lin+"_"+level)
            else:
                trim_lin = 'Unknown_' + level
                col = headers.index(trim_lin)
            for i in range(0, len(data)):
                data[i][col] = data[i][col] + float(line[i+1])
        ix = ix + 1
    f.close()
    df = pd.DataFrame(data)
    df.columns = headers
    return df
    
def getPerioData(random_seed, taxa_levels):
    file = 'perio_data.txt'
    taxa_order = ['domain', 'phylum', 'class', 'order', 'family', 'genus', 'species']
    taxas = getTaxa(file)
    taxa_dict = {}
    taxa_level_dict = {}
    for lev in taxa_levels:
        taxa_dict[lev] = taxas[lev]
        taxa_level_dict[lev] = taxa_order.index(lev)
    data = loadData(file, taxa_dict, taxa_level_dict)
    for lev in taxa_levels:
        data.loc[:,taxas[lev][0]+"_"+lev:("Unknown_" +lev)] = data.loc[:,taxas[lev][0]+"_"+lev:"Unknown_" +lev].div(data.loc[:,taxas[lev][0]+"_"+lev:("Unknown_" +lev)].sum(axis=1), axis=0)
    return data
    
###Main###

file = 'perio_data.txt'

'''
random_seed = 4
taxa_order = ['domain', 'phylum', 'class', 'order', 'family', 'genus', 'species']
taxa_level = 'phylum'

#Load in data, taxas first and then sample data
taxas = getTaxa(file)
#print (taxas['phylum'])
data = loadData(file, taxas[taxa_level], taxa_order.index(taxa_level))

#Scale to percentages
#print (data[:1])
data.loc[:,taxas[taxa_level][0]:"Unknown"] = data.loc[:,taxas[taxa_level][0]:"Unknown"].div(data.sum(axis=1), axis=0)
#print (data[1:2])

#TODO make sure I handle the case where I want to use cross-validation instead of a validation set
#TODO what if I want to remove features?

feature_class_split = splitScaleSeparate(data, random_seed)

#print (feature_class_split['test']['features'][:1])
#print (feature_class_split['test']['classes'][:1])
'''
a=2
