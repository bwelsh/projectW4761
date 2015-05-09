import sys
import json
import numpy as np
import pandas as pd
import math
from cgen_include import *

###Functions###
    
def loadSubjectMapping(file):
    '''
    Given a file, this maps the samples to the ids used in the data file
    '''
    f = open(file)
    mapping = {}
    for line in f:
        line = line.strip()
        if line != '':
            subj, id = line.split('\t')
            if subj[:2] != 'MH':
                mapping[id] = subj
    return mapping

def loadData(file, mapping):
    '''
    Given a file and subject mapping, this function aggregates the data on the reads by parsing the given lineages and counting the number of each type for each subject and storing it in a dict. Two other dictionaries are created with various taxonomic relationships in it, but turned out not to be needed in the implementation.
    '''
    f = open(file)
    taxa_order = ['root', 'cell', 'domain', 'phylum', 'class', 'order', 'family', 'genus', 'species']
    taxonomy = {}
    subj_dict = {}
    tax_categories = {'domain': [], 'phylum': [], 'class': [], 'order': [], 'family': [], 'genus': [], 'species': []}
    for line in f:
        line = line.strip()
        if line != '':
            overall_data = line.split('_')
            subject = overall_data[1]
            if subject[:2] != 'MH':
                if line.split('\t')[0] in mapping:
                    subject = mapping[line.split('\t')[0]]
            if subject not in subj_dict:
                subj_dict[subject] = {'total': 0, 'domain': {}, 'phylum': {}, 'class': {}, 'order': {}, 'family': {}, 'genus': {}, 'species': {}}
            overall_tax = overall_data[-1]
            if '+' in overall_tax:
                all_tax = overall_tax[overall_tax.find('+'):].strip()
            elif '-' in overall_tax:
                all_tax = overall_tax[overall_tax.find('-'):].strip()
            else:
                all_tax = 'NA'
            if ';' in all_tax:
                taxas = all_tax.split(';')
            else:
                taxas = [all_tax]
            subj_dict[subject]['total'] += 1
            if len(taxas) > 2:
                for i in range(2, len(taxas)):
                    if taxas[i] in subj_dict[subject][taxa_order[i]]:
                        subj_dict[subject][taxa_order[i]][taxas[i]] += 1
                    else:
                        subj_dict[subject][taxa_order[i]][taxas[i]] = 1
                    if i < len(taxas)-1:
                        if taxas[i] in taxonomy:
                            if taxas[i+1] not in taxonomy[taxas[i]]:
                                taxonomy[taxas[i]].append(taxas[i+1])
                        else:
                            taxonomy[taxas[i]] = [taxas[i+1]]
                    if taxas[i] not in tax_categories[taxa_order[i]]:
                        tax_categories[taxa_order[i]].append(taxas[i])
            if len(taxas) > 9:
                print (taxas)
    f.close()
    return subj_dict, taxonomy, tax_categories
    
def writeProcessedData(data, out_file_name):
    '''
    Given a dictionary and a file name, write this dictionary to the filename provided in json format.
    '''
    with open(out_file_name, 'w') as out_file:
        json.dump(data, out_file)
        
    
###Main###

in_file = 'UniqGene_NR.tax.catalog'
map_file = 'Gene2Sample.list'
diagnosis_file = 'catalog_diagnosis.txt'

#This dataset is a little large. To avoid needing to keep working with it each time, this code takes the data and saves the relevant summary data (percentage of reads found for each feature) to a file
md = loadSubjectMapping(map_file)
sub_data, taxonomy, tax_categories = loadData(in_file, md)
writeProcessedData(sub_data, 'sub_data.json')
writeProcessedData(taxonomy, 'taxonomy.json')
writeProcessedData(tax_categories, 'tax_categories.json')
