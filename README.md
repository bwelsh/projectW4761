# projectW4761
##System Requirements

This analysis was run in Python 3.4, using additional packages numpy 1.8.2, pandas 0.16.0, matplotlib 1.3.1, and sklearn 0.16.1. It was run on a standard laptop, although a more powerful computer might cut down on the pre-processing time for some files. 

##File Descriptions

####Pre-process files
cgen_naive.py - This file contains functions for pre-processing the dataset from the article, "The Treatment-Naive Microbiome in New-Onset Crohn’s Disease". The data comes from Supplementary Table 2 of this article, tab Table2I-key taxa, saved and read in as "samples_1040.csv" in this repository. The main function to run is "getOrigSplitData(random_seed, levels)", where the random seed a value selected for the entire project in order to maintain consistency of results and levels is a list of the taxa level(s) to gather. For this dataset, the options are 'phylum', 'order', or both. Due to the skew in this dataset, the train/validate/test split was implemented by hand (to ensure roughly even health vs disease samples in the training set) and so this functions returns the data and the levels requested split into train/validate/test sets. The function returns a dictionary with keys train, valid, and test, each of which has keys for features (the feature dataframe), classes (the ground truth diagnosis dataframe for each sample), and feature_cols (a list of the features for this dataset). This function is run in both cgen_params_select.py and cgen_project.py.

cgen_pre_catalog.py - This file contains functions for the initial pre-processing of the dataset from the article, "A human gut microbial gene catalogue established by metagenomic sequencing". The data comes from http://gutmeta.genomics.org.cn/, referenced in this article. Two files from this site were used - "UniqGene_NR.tax.catalog" and "Gene2Sample.list". They are not included here due to their size. Running the function "loadSubjectMapping(Gene2Sample.list)" and then "loadData(UniqGene_NR.tax.catalog, [result of loadSubjectMapping])" produces three dictionaries that can then be written out to json files using the function "writeProcessedData(data, out_file_name)". This was done to speed processing time. The three produced files are included in the repository and are named "sub_data.json", "taxonomy.json", and "tax_categories.json". In addition, a "catalog_diagnosis.txt" file was created manually to contain the class information (BMI) for each sample, from Supplementary Table 1 of this article. 

cgen_catalog.py - This file contains functions for the secondary pre-processing of the dataset from the article, "A human gut microbial gene catalogue established by metagenomic sequencing". The main function to run to get this data for analysis is "getCatalogData(random_seed, taxa_levels, disease_bmi_threshold)", where the random seed a value selected for the entire project in order to maintain consistency of results, taxa_levels is a list of the taxa level(s) to gather, and disease_bmi_threshold is the threshold to distinguish healthy from diseased samples. In this analysis, 30 was chosen as the threshold, but could be changed by passing in a different value when getting the data. For this dataset, the taxa level(s) to choose from are 'domain', 'phylum', 'class', 'order', 'family', 'genus', or 'species', or any combination of those. Returned from this function is a dataframe with a sample column, a diagnosis column, and feature columns representing the frequencies of each taxanomic class for the taxa levels passed in. This function is run in both cgen_params_select.py and cgen_project.py.

cgen_pre_richness.py - This file contains functions for pre-processing the dataset from the article, "Richness of human gut microbiome correlates with metabolic markers". The data comes from the Supplementary Data excel file, SupTab 4 and SupTab6. These tabs have been saved as "richness_data1.txt" and "richness_data2.txt" for processing, and both have been placed in this repository. The main function to run is "getRichnessData(random_seed, taxa_levels, disease_bmi_threshold)", where the random seed a value selected for the entire project in order to maintain consistency of results, taxa_levels is a list of the taxa level(s) to gather, and disease_bmi_threshold is the threshold to distinguish healthy from diseased samples. In this analysis, 30 was chosen as the threshold, but could be changed by passing in a different value when getting the data. For this dataset, the taxa level(s) to choose from are 'domain', 'phylum', 'class', 'order', 'family', or 'genus', or any combination of those. Returned from this function is a dataframe with a sample column, a diagnosis column, and feature columns representing the frequencies of each taxanomic class for the taxa levels passed in, along with a feature column listing the total number of genes for this sample. This function is run in both cgen_params_select.py and cgen_project.py.

cgen_pre_perio.py - This file contains function for pre-processing the dataset from the article, "Bacterial Community Composition of Chronic periodontitis and novel oral sampling sites for detecting disease indicators". The data comes from Additional File 7, the Pooled tab. This tab has been saved as "perio_data.txt" and has been placed in this repository. The main function to run is "getPerioData(random_seed, taxa_levels)", where the random seed a value selected for the entire project in order to maintain consistency of results and taxa_levels is a list of the taxa level(s) to gather. For this dataset, the taxa level(s) to choose from are 'domain', 'phylum', 'class', 'order', 'family', 'genus', or 'species', or any combination of those. Returned from this function is a dataframe with a sample column, a diagnosis column, and feature columns representing the frequencies of each taxanomic class for the taxa levels passed in. This function is run in both cgen_params_select.py and cgen_project.py.

####Input files
catalog_diagnosis.txt - input file for class information for the cgen_catalog.py file

perio_data.txt - input file for the cgen_pre_perio.py file

richness_data1.txt - input file for the cgen_pre_richness.py file

richness_data2.txt - input file for the cgen_pre_richness.py file

samples_1040.csv - input file for the cgen_naive.py file

sub_data.json - output from pre-processing in cgen_catalog.py, used as input for the cgen_catalog.py file

tax_categories.json - output from pre-processing in cgen_catalog.py

taxonomy.json - output from pre-processing in cgen_catalog.py

####Analysis files
cgen_include.py - This file contains functions used in both the cgen_param_select.py file and the cgen_project.py file in order to support the processing and analysis of the data. Functions include those needed to split the data into train/validate/split, scale the data, separate the features from the classes, add polynomial features, perform feature selection, and assess the fit of the model to the data.

cgen_param_select.py - This file contains the models used in this analysis, along with the parameters to test in the crossvalidation grid-search to find the optimal parameters. This was separated from the main project file due to processing time. Running this file for all studied data takes on the order of 8 hours on a standard laptop. This file sets up all taxa levels to use as features (for this analysis, all combinations up to three taxa levels were trained), and given the taxa levels, reads in data from the pre-processing functions listed above, then runs a grid search for each model for the parameter values given to find the best parameters for each model, for each dataset, for each combination of taxa levels, along with the same for the combined dataset (a combination of all data sources). It then creates a dictionary of all of these results and writes them to files titled "fit#.json", where '#' is the number of the file. These fit files are included in this repository and are used as the input to the main project file.

fit1.json - best parameters for models including 2 taxanomic levels

fit2.json - best parameters for models including 1 taxanomic level

fit3.json - best parameters for models including 3 taxanomic levels for the perio dataset

fit4.json - best parameters for models including 3 taxanomic levels for the richness dataset

fit5.json - best parameters for models including 3 taxanomic levels for the catalog dataset

cgen_project.py - This files contains the functions for the main analysis and the production of plots and tables. It reads in the fit files listed above and then performs various analysis. The results of the analysis are output to the files listed below in the Result files section.

####Result files
best_all.png - This plot is created by the cgen_project.py file. It plots the ROC space for validation and test data for the best model for the best levels for each dataset.

best_level_catalog.png - This plot is created by the cgen_project.py file. It plots the ROC space for validation and test data for the best parameters for each model for the best levels of the catalog dataset.

best_level_multi.png - This plot is created by the cgen_project.py file. It plots the ROC space for validation and test data for the best parameters for each model for the best levels of the combined dataset.

best_level_naive.png - This plot is created by the cgen_project.py file. It plots the ROC space for validation and test data for the best parameters for each model for the best levels of the naive dataset.

best_level_perio.png - This plot is created by the cgen_project.py file. It plots the ROC space for validation and test data for the best parameters for each model for the best levels of the perio dataset.

best_level_richness.png - This plot is created by the cgen_project.py file. It plots the ROC space for validation and test data for the best parameters for each model for the best levels of the richness dataset.

cross_predict_catalog.png - This plot is created by the cgen_project.py file. It plots the sensitivity plus specificity score for each technique for the test set of each dataset, with the models trained on the Catalog dataset.

cross_predict_naive.png - It plots the sensitivity plus specificity score for each technique for the test set of each dataset, with the models trained on the Naive dataset.

cross_predict_perio.png - It plots the sensitivity plus specificity score for each technique for the test set of each dataset, with the models trained on the Perio dataset.

cross_predict_richness.png - It plots the sensitivity plus specificity score for each technique for the test set of each dataset, with the models trained on the Richness dataset.

feature_importances.txt - This file is created by the cgen_project.py file. It gives the feature importances in descending order for the best forest found for each dataset.

mean_score_technique.png - This file is created by the cgen_project.py file. It plots the average score for each model for each of the datasets.

median_score_taxa.png - This file is created by the cgen_project.py file. It plots the median score for each taxa level for each of the datasets.

##Running the Project

To run this project successfully, all files must be in the same directory. Here is the order to run the files in:

1. Run cgen_pre_catalog.py. This will create "sub_data.json", which is needed for pre-processing the catalog data.

2. Run cgen_param_select.py. This will find the best fit for each model for each taxa level(s) (up to 3) and output the results to "fit1.json", "fit2.json", "fit3.json", "fit4.json" and "fit5.json", for use in analysis. It uses functions from cgen_include.py and the cgen preprocessing files to complete this task. Note: this will take hours to run on a standard laptop. The smallest file, "fit2.py", which calculates the best model for a single taxanomic level, can be created in under an hour if desired. To do this, comment out lines: aggregateBestParameters(datasets, parameters, random_seed, [2], 1), aggregateBestParameters(p_dataset, parameters, random_seed, [3], 3), aggregateBestParameters(r_dataset, parameters, random_seed, [3], 4), and aggregateBestParameters(c_dataset, parameters, random_seed, [3], 5) before running this file.

3. Run cgen_project.py. This will complete the analysis of the data and create the plots in this repository (although they must be saved manually). It should run in under 10 minutes, due to the time-consuming parameter selection being done in the step above and saved to files.
