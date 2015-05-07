import numpy as np
import pandas as pd
import math
from cgen_include import *

###Functions###

def getTaxa(file):
    taxa_order = ['genus', 'family', 'order', 'class', 'phylum', 'domain']
    tax_categories = {'domain': [], 'phylum': [], 'class': [], 'order': [], 'family': [], 'genus': []}
    clusters = {}
    f = open(file)
    count = 0
    for line in f:
        if count > 0:
            line = line.strip()
            line = line.split('\t')
            clust = line[0].strip()
            clusters[clust] = {}
            lineage = line[1:]
            if len(lineage) == 0:
                for i in range(0, len(taxa_order)):
                    clusters[clust][taxa_order[i]] = 'Unknown_'+taxa_order[i]
            for i in range(0, len(lineage)):
                if lineage[i] != '':
                    tax_categories[taxa_order[i]].append(lineage[i].strip()+'_'+taxa_order[i])
                    clusters[clust][taxa_order[i]] = lineage[i].strip()+'_'+taxa_order[i]
                else:
                    clusters[clust][taxa_order[i]] = 'Unknown_'+taxa_order[i]
        count += 1
    f.close()
    for cat in tax_categories:
        tax_categories[cat] = set(tax_categories[cat])
        tax_categories[cat] = list(tax_categories[cat])
        tax_categories[cat].append('Unknown_'+cat)
    return tax_categories, clusters
    
def loadData(file, cats, cluster_map, level, diagnosis_threshold):
    data = []
    headers = ['Sample', 'Diagnosis', 'Num_Genes']
    for lev in cats:
        for item in cats[lev]:
            headers.append(item)
    file_headers = []
    ix = 0
    f = open(file)
    for line in f:
        if ix == 0:
            file_headers = line.strip().split('\t')
        else:
            sample_data = {}
            for lev in cats:
                for cat in cats[lev]:
                    sample_data[cat] = 0
            line = line.strip()
            line = line.split('\t')
            for i in range(0, len(line)):
                if file_headers[i].strip() == 'Individual':
                    sample_data['Sample'] = line[i].strip()
                elif file_headers[i].strip() == 'Gene Nb':
                    sample_data['Num_Genes'] = int(math.floor(float(line[i].strip())))
                elif file_headers[i].strip() == 'BMI':
                    if float(line[i].strip()) > diagnosis_threshold: 
                        sample_data['Diagnosis'] = 1
                    else:
                        sample_data['Diagnosis'] = -1
                else:
                    for lev in cats:
                        category = cluster_map[file_headers[i].strip()][lev]
                        sample_data[category] = sample_data[category] + float(line[i].strip())
            data.append(sample_data)
        ix = ix + 1
    f.close()
    df = pd.DataFrame(data)
    return df
    
def getRichnessData(random_seed, taxa_levels, disease_bmi_threshold):
    file = 'richness_data1.txt'
    sample_file = 'richness_data2.txt'
    taxas, cluster_map = getTaxa(file)
    taxa_dict = {}
    for lev in taxa_levels:
        taxa_dict[lev] = taxas[lev]
    data = loadData(sample_file, taxa_dict, cluster_map, taxa_levels, disease_bmi_threshold)
    return data
