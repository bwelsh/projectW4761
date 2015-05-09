import sys
import json
import numpy as np
import pandas as pd
import math
from cgen_include import *

###Functions###

def loadProcessedData(in_file_name):
    '''
    Given a files name, use the json parser to read in the data and return it
    '''
    with open(in_file_name) as in_file:
        in_data = json.load(in_file)
    return in_data
    
def loadDiseaseDict(file, threshold):
    '''
    Given the filename and a threshold, read in the data on disease status (either IBD/Not or BMI and if BMI, use threshold to determine disease status) and return a dictionary with the sample as the key and the disease status (1 for disease, -1 for healthy) as the value
    '''
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
    '''
    Given a dictionary of sample data, create a dictionary of lists of microbial names at each taxa level that appear in the dataset
    '''
    tax_categories = {'domain': [], 'phylum': [], 'class': [], 'order': [], 'family': [], 'genus': [], 'species': []}
    for subject in subject_data:
        for taxa in subject_data[subject]:
            if taxa != 'total':
                for tax_cat in subject_data[subject][taxa]:
                    tax_categories[taxa].append(tax_cat+'_'+taxa)
    for cat in tax_categories:
        tax_categories[cat] = set(tax_categories[cat])
        tax_categories[cat] = list(tax_categories[cat])
        tax_categories[cat].append('Unknown_' + cat)
    return tax_categories
    
def collateData(sub_data, taxas, taxa_levels, disease_status):
    '''
    Given sample data, a dictionary of all of the microbial names at each taxa levels, a list of taxa levels to include, and the disease_status dictionary, create a dataframe with the features (microbial names) for the taxa levels passed in and fill in the values for each sample, along with the diseases status for each sample
    '''
    data = []
    headers = ['Sample', 'Diagnosis', 'Num_Genes']
    for lev in taxa_levels:
        for item in taxas[lev]:
            headers.append(item)
    for subject in sub_data:
        if subject != 'unmapped': #Some reads were not mapped to a sample, we need to ignore these
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
                    hd, tx = head.split('_')
                    if hd in sub_data[subject][taxa_level]:
                        ln_data[head] = float(sub_data[subject][taxa_level][hd]) / sub_data[subject]['total']
                    else:
                        ln_data[head] = 0
            data.append(ln_data)
    return pd.DataFrame(data)
    
def getCatalogData(random_seed, taxa_levels, disease_bmi_threshold):
    '''
    The "public" interface to this file, what the other files call in order to get the data for this dataset. The taxa levels desired are required, along with the disease_bmi_threshold, in order to determine whether a sample is considered diseased or healthy, based on the bmi.
    '''
    sub_data = loadProcessedData('sub_data.json')
    diagnosis_file = 'catalog_diagnosis.txt'
    taxas = getTaxa(sub_data)
    disease_mapping = loadDiseaseDict(diagnosis_file, disease_bmi_threshold)
    data = collateData(sub_data, taxas, taxa_levels, disease_mapping)
    return data 
