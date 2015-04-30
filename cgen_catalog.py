import sys
import json
import numpy as np
import pandas as pd
import math
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from cgen_include import *


###Functions###

def subsetData(in_file, out_file, subject):
    f = open(in_file)
    subset = ''
    count = 0
    for line in f:
        if line.strip().split('_')[1] == subject:
            subset = subset + line + '\n'
        if count%100000 == 0:
            print (count)
        count = count + 1
    f.close()
    g = open(out_file, 'w')
    g.write(subset)
    g.close()
    
def getSubjects(in_file):
    f = open(in_file)
    subjects = []
    for line in f:
        if line.strip().split('_')[1] not in subjects:
            subjects.append(line.strip().split('_')[1])
    f.close()
    return subjects
    
def loadSubjectMapping(file):
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
    with open(out_file_name, 'w') as out_file:
        json.dump(data, out_file)
        
def loadProcessedData(in_file_name):
    with open(in_file_name) as in_file:
        in_data = json.load(in_file)
    return in_data
    
def printDiversity(tax_categories):
    for cat in tax_categories:
        print (cat, ':', len(tax_categories[cat]))
        
def countTaxas(category, level, levels_down, sub_data, taxonomy):
    print('Subject has', len(sub_data[level]), 'out of', len(taxonomy[category]), level)
    print(sub_data[level])
    print(taxonomy[category])
    #for i in range(0, levels_down):
    
def loadDiseaseDict(file, threshold):
    f = open(file)
    mapping = {}
    headers = []
    ix = 0
    for line in f:
        line = line.strip()
        line = line.split(' ')
        if ix == 0:
            headers = line
        else:
            if line != '':
                subj = line[headers.index('Sample')]
                if subj[:2] != 'MH':
                    status = line[headers.index('IBD')]
                    if status == 'Y':
                        mapping[subj] = 1
                    else:
                        mapping[subj] = -1
                else:
                    bmi = float(line[headers.index('BMI')])
                    if bmi > threshold:
                        mapping[subj] = 1
                    else:
                        mapping[subj] = -1
        ix += 1
    return mapping
    
def getTaxa(subject_data):
    tax_categories = {'domain': [], 'phylum': [], 'class': [], 'order': [], 'family': [], 'genus': [], 'species': []}
    for subject in subject_data:
        for taxa in subject_data[subject]:
            if taxa != 'total':
                for tax_cat in subject_data[subject][taxa]:
                    tax_categories[taxa].append(tax_cat)
    for cat in tax_categories:
        tax_categories[cat] = set(tax_categories[cat])
        tax_categories[cat] = list(tax_categories[cat])
        tax_categories[cat].append('Unknown_' + cat)
    return tax_categories
    
def collateData(sub_data, taxas, taxa_levels, disease_status):
    data = []
    headers = ['Sample', 'Diagnosis', 'Num_Genes']
    for lev in taxa_levels:
        for item in taxas[lev]:
            headers.append(item)
    for subject in sub_data:
        #TODO fix this? we don't want reads from unmapped subjects
        if subject != 'unmapped':
            ln_data = {}
            for head in headers:
                if head == 'Sample':
                    ln_data[head] = subject
                elif head == 'Diagnosis':
                    ln_data[head] = disease_status[subject]
                elif head == 'Num_Genes':
                    ln_data[head] = sub_data[subject]['total']
                elif head[:7] == 'Unknown':
                    taxa_level = head[8:]
                    ln_data[head] = float(sub_data[subject]['total'] - sum(sub_data[subject][taxa_level].values())) / sub_data[subject]['total']
                else:
                    taxa_level = ''
                    for tax in taxas:
                        if head in taxas[tax]:
                            taxa_level = tax
                    if head in sub_data[subject][taxa_level]:
                        ln_data[head] = float(sub_data[subject][taxa_level][head]) / sub_data[subject]['total']
                    else:
                        ln_data[head] = 0
            data.append(ln_data)
    return pd.DataFrame(data)
    
def getCatalogData(random_seed, taxa_levels, disease_bmi_threshold):
    sub_data = loadProcessedData('sub_data.json')
    diagnosis_file = 'catalog_diagnosis.txt'
    
    #TODO do I want to add a feature for the unique counts by taxa (like unique species)?
    #TODO feature selection even at the phylum level?
    
    taxas = getTaxa(sub_data)
    disease_mapping = loadDiseaseDict(diagnosis_file, disease_bmi_threshold)
    data = collateData(sub_data, taxas, taxa_levels, disease_mapping)
    return data 
    
###Main###

in_file = 'UniqGene_NR.tax.catalog'
map_file = 'Gene2Sample.list'
out_file = 'MH0022_subset.txt'
subject = 'V1.CD-12'
test_file = 'test_tax.txt'
diagnosis_file = 'catalog_diagnosis.txt'


#subsetData(in_file, out_file, subject)
#sub_data, taxonomy, tax_categories = loadData(in_file)
#print (len(sub_data[subject]['species'].keys()))
#print (sub_data[subject]['total'])
#print (taxonomy['Bacteroidetes'])
#countTaxas('Bacteria', 'phylum', 1, sub_data, taxonomy)

#subjects = getSubjects(in_file)
#print (subjects)
#print (len(subjects))

#Code to run initially to process the raw data
'''
md = loadSubjectMapping(map_file)
sub_data, taxonomy, tax_categories = loadData(in_file, md)
writeProcessedData(sub_data, 'sub_data.json')
writeProcessedData(taxonomy, 'taxonomy.json')
writeProcessedData(tax_categories, 'tax_categories.json')
'''

#Code to load the data back in when we want to work with it


#print (feature_class_split['test']['features'][:1])
#print (feature_class_split['test']['classes'][:1])

#print (data[100:101])
#print (taxas['domain'])
#print (sub_data[subject]['total'])
#print (sum(sub_data[subject]['domain'].values()))
#print (sub_data[subject].keys())
