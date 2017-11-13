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
    
    habitable_slice = habitable[int(habitable_size*start):int(habitable_size*(start+width))];    
    non_habitable_slice = non_habitable[int(non_habitable_size*start):int(non_habitable_size*(start+width))];
    
    habitable_slice_features = np.ones(len(habitable_slice));    
    non_habitable_slice_features = np.full((len(non_habitable_slice)), -1);
    
    for i in feature_indexes:
        habitable_slice_features = np.column_stack((habitable_slice_features,  habitable_slice[i]));        
        non_habitable_slice_features = np.column_stack((non_habitable_slice_features , non_habitable_slice[i]));
        
    X = np.vstack((habitable_slice_features[:,1:], non_habitable_slice_features[:,1:])) ;
    Y = np.append(habitable_slice_features[:,0], non_habitable_slice_features[:,0]);
    
    return X, Y;

def get_test_error(habitable_planets, non_habitable_planets, features, train_slice, test_slice):
    X_train, Y_train = get_selected_features(habitable, non_habitable, features, 0.0, train_slice);
    X_dev, Y_dev = get_selected_features(habitable, non_habitable, features, train_slice, dev_slice);
                
    clf = svm.SVC(kernel='rbf', gamma=10)
    clf.fit(X_train, Y_train)
        
    y_predicted = clf.predict(X_dev);
        
    result = y_predicted * Y_dev;
        
    error = (sum(1 for i in result if i <= 0)/len(Y_dev))*100;
    
    return error;
    
    
def select_best_features (habitable, non_habitable, train_slice, dev_slice):
    selected_features = set([]);
    previous_min_error = 0;
    for i in planetary_stellar_parameter_cols:
        min_error = 100;        
        min_index = 0;
        for j in planetary_stellar_parameter_cols:
            if j not in selected_features:
                tmp_selected_features = set(selected_features);
                tmp_selected_features.add(j);
                
                error = get_test_error(habitable, non_habitable, tmp_selected_features, train_slice, dev_slice);
                
                if error < min_error:
                    min_index = j;
                    min_error = error;
        
        if previous_min_error == min_error:
            break;
            
        selected_features.add(min_index);
        previous_min_error = min_error;
    
    print('Selected features = ', selected_features, ' with min error ', previous_min_error);
    return selected_features;


#Index is 0 based

# Planetary and Stellar parameters    
planetary_stellar_parameter_indexes = (15,  # koi period,      Orbital Period [days]
                                       42,  # koi_ror:         Planet-Star Radius Ratio
                                       45,  # koi_srho:        Fitted Stellar Density [g/cm**3] -
                                       49,  # koi_prad:        Planetary Radius [Earth radii]
                                       52,  # koi_sma:         Orbit Semi-Major Axis [AU]
                                       58,  # koi_teq:         Equilibrium Temperature [K]
                                       61,  # koi_insol:       Insolation Flux [Earth flux]
                                       64,  # koi_dor:         Planet-Star Distance over Star Radius
                                       76,  # koi_count:       Number of Planet 
                                       87,  # koi_steff:       Stellar Effective Temperature [K] 
                                       90,  # koi_slogg:       Stellar Surface Gravity [log10(cm/s**2)]
                                       93,  # koi_smet:        Stellar Metallicity [dex]
                                       96,  # koi_srad:        Stellar Radius [Solar radii]
                                       99   # koi_smass:       Stellar Mass [Solar mass]
                                       );
#Names of columns from kepler data
planetary_stellar_parameter_cols = (   "koi_period",    # koi_period       Orbital Period [days]
                                       "koi_ror",       # koi_ror:         Planet-Star Radius Ratio
                                       "koi_srho",      # koi_srho:        Fitted Stellar Density [g/cm**3] -
                                       "koi_prad",      # koi_prad:        Planetary Radius [Earth radii]
                                       "koi_sma",       # koi_sma:         Orbit Semi-Major Axis [AU]
                                       "koi_teq",       # koi_teq:         Equilibrium Temperature [K]
                                       "koi_insol",     # koi_insol:       Insolation Flux [Earth flux]
                                       "koi_dor",       # koi_dor:         Planet-Star Distance over Star Radius
                                       "koi_count",     # koi_count:       Number of Planet 
                                       "koi_steff",     # koi_steff:       Stellar Effective Temperature [K] 
                                       "koi_slogg",     # koi_slogg:       Stellar Surface Gravity [log10(cm/s**2)]
                                       "koi_smet",      # koi_smet:        Stellar Metallicity [dex]
                                       "koi_srad",      # koi_srad:        Stellar Radius [Solar radii]
                                       "koi_smass"      # koi_smass:       Stellar Mass [Solar mass]
                                       );
                                       
try:    
    habitable_planets = np.genfromtxt('../data/habitable_planets_detailed_list.csv',filling_values = 0, names=True, dtype=float, delimiter=",",usecols=planetary_stellar_parameter_indexes);
    
    non_habitable_planets = np.genfromtxt('../data/non_habitable_planets_confirmed_detailed_list.csv', filling_values = 0, names = True, dtype=float, delimiter=",",usecols=planetary_stellar_parameter_indexes);
    
    train_slice = 0.5;
    dev_slice = 0.2;
    
    best_features = select_best_features(habitable_planets, non_habitable_planets, train_slice, dev_slice);
    
    #now train on larger slice (train+dev) with given feature and run test on remaining data
    train_slice = train_slice+dev_slice;
    test_slice = 1.0 - train_slice;
    test_error = get_test_error(habitable_planets, non_habitable_planets, best_features, train_slice, test_slice);
    
    print('Test error on trained data is ', test_error);

except ValueError:
    print('Error reading file');
    raise;

    