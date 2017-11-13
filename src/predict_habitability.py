#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 21:14:12 2017

@author: ramisra
"""

import numpy as np
from sklearn import svm         

def get_selected_features(habitable, non_habitable, feature_indexes, start, width):
    
    habitable_size = len(habitable);
    non_habitable_size = len(non_habitable);
    
    habitable_slice = habitable[int(habitable_size*start):int(habitable_size*(start+width)),:];    
    non_habitable_slice = non_habitable[int(non_habitable_size*start):int(non_habitable_size*(start+width)),:];
    
    habitable_slice_features = np.ones(len(habitable_slice));    
    non_habitable_slice_features = np.full((len(non_habitable_slice)), -1);
    
    for i in feature_indexes:
        habitable_slice_features = np.column_stack((habitable_slice_features,  habitable_slice[:,i]));        
        non_habitable_slice_features = np.column_stack((non_habitable_slice_features , non_habitable_slice[:,i]));
        
    X = np.vstack((habitable_slice_features[:,1:], non_habitable_slice_features[:,1:])) ;
    Y = np.append(habitable_slice_features[:,0], non_habitable_slice_features[:,0]);
    
    return X, Y;
    
    
def select_best_features (habitable, non_habitable, train_slice, dev_slice):
    included_indexes = habitable.shape[1];
    selected_indexes = set([]);
    previous_min_error = 0;
    for i in range(included_indexes):
        min_error = 100;        
        min_index = 0;
        for j in range(included_indexes):
            if j not in selected_indexes:
                tmp_selected_features = set(selected_indexes);
                tmp_selected_features.add(j);
                
                X_train, Y_train = get_selected_features(habitable, non_habitable, tmp_selected_features, 0.0, train_slice);
                X_dev, Y_dev = get_selected_features(habitable, non_habitable, tmp_selected_features, train_slice, dev_slice);
                
                clf = svm.SVC(kernel='rbf', gamma=10)
                clf.fit(X_train, Y_train)
        
                y_predicted = clf.predict(X_dev);
        
                result = y_predicted * Y_dev;
        
                error = (sum(1 for i in result if i <= 0)/len(Y_dev))*100;
                
                if error < min_error:
                    min_index = j;
                    min_error = error;
        
        if previous_min_error == min_error:
            break;
            
        selected_indexes.add(min_index);
        previous_min_error = min_error;
    
    print('selected_indexes = ', selected_indexes, ' min error ', previous_min_error);
    return selected_indexes;

#Index is 0 based

# Planetary and Stellar parameters    
planetary_stellar_parameter_indexes = (15,  # koi period,      Orbital Period [days]
                                       30,  # koi_impact:      Impact Parameter
                                       39,  # koi_depth:       Transit Depth [ppm]
                                       42,  # koi_ror:         Planet-Star Radius Ratio
                                       45,  # koi_srho:        Fitted Stellar Density [g/cm**3] -
                                       49,  # koi_prad:        Planetary Radius [Earth radii]
                                       52,  # koi_sma:         Orbit Semi-Major Axis [AU]
                                       55,  # koi_incl:        Inclination [deg]
                                       58,  # koi_teq:         Equilibrium Temperature [K]
                                       61,  # koi_insol:       Insolation Flux [Earth flux]
                                       64,  # koi_dor:         Planet-Star Distance over Star Radius
                                       76,  # koi_count:       Number of Planet 
                                       84,  # koi_model_chisq: Chi-Square
                                       87,  # koi_steff:       Stellar Effective Temperature [K] 
                                       90,  # koi_slogg:       Stellar Surface Gravity [log10(cm/s**2)]
                                       93,  # koi_smet:        Stellar Metallicity [dex]
                                       96,  # koi_srad:        Stellar Radius [Solar radii]
                                       99,  # koi_smass:       Stellar Mass [Solar mass]
                                       102);# koi_sage:        Stellar Age [Gyr]
    
habitable = np.loadtxt(open('../data/habitable_planets_detailed_list.csv'), delimiter=",",usecols=planetary_stellar_parameter_indexes);

non_habitable = np.loadtxt(open('../data/non_habitable_planets_confirmed_detailed_list.csv'), delimiter=",",usecols=planetary_stellar_parameter_indexes);

train_slice = 0.5;
dev_slice = 0.2;

best_features = select_best_features(habitable, non_habitable, train_slice, dev_slice);
    