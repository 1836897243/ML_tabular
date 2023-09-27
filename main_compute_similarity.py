from Analyse import *
from LoaderContainer import LoaderContainer
import numpy as np
import pandas as pd
import os
import ast
root_dir = 'dataset/regression'

batch_size = 128
shuffle = False
datasets = os.listdir(root_dir)
for dataset_name in datasets:
    cur_dir = os.path.join(root_dir, dataset_name)
    if(os.path.isdir(cur_dir)) and 'Data' in os.listdir(cur_dir):
        print(cur_dir)
        loaderContainer = LoaderContainer(cur_dir + '/', batch_size, shuffle)
        analyse = Analyse(loaderContainer)
        cur_dir = os.path.join(cur_dir, 'Data')
        for file in os.listdir(cur_dir):
            file_name = os.path.join(cur_dir, file)
            DF = pd.read_excel(file_name)

            mean_degrees = [analyse.compute_mean_degree_of_numerical_features(ast.literal_eval(feature_list))
                            for feature_list in DF['feature']]
            DF['mean_degree'] = mean_degrees

            DistanceCorrelations = [analyse.compute_distance_correlation(ast.literal_eval(feature_list))
                                    for feature_list in DF['feature']]
            DF['DistanceCorrelation'] = DistanceCorrelations

            DF.to_excel(file_name,index=False)