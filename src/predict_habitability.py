#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ramisra
"""

import sys
import numpy as np
from sklearn import svm         

TRAIN_DATA = 0.5;
DEV_DATA = 0.2;
BEST_FEATURE_SELECTION_LOOP_COUNT=30;

#Index is 0 based

# Planetary and Stellar parameters    
planetary_stellar_parameter_indexes = (2,   # kepoi_name:      KOI Name
                                       15,  # koi period,      Orbital Period [days]
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

def select_features(from_data, to_data, feature_indexes):
    for i in feature_indexes:
        to_data = np.column_stack((to_data, from_data[i]));
        
    return to_data;    
                                    
def get_X_Y(habitable, non_habitable, feature_indexes, start, width):
    
    habitable_size = len(habitable);
    non_habitable_size = len(non_habitable);
    
    habitable_slice = habitable[int(habitable_size*start):int(habitable_size*(start+width))];    
    non_habitable_slice = non_habitable[int(non_habitable_size*start):int(non_habitable_size*(start+width))];
    
    habitable_slice_features = np.ones(len(habitable_slice));    
    non_habitable_slice_features = np.full((len(non_habitable_slice)), -1);
    
    habitable_slice_features = select_features(habitable_slice, habitable_slice_features, feature_indexes);
    non_habitable_slice_features = select_features(non_habitable_slice, non_habitable_slice_features, feature_indexes);
    
    X = np.vstack((habitable_slice_features[:,1:], non_habitable_slice_features[:,1:])) ;
    Y = np.append(habitable_slice_features[:,0], non_habitable_slice_features[:,0]);
    
    return X, Y;

def do_svm(X_train, Y_train, X_predict):
    clf = svm.SVC(kernel='rbf', gamma=10)
    clf.fit(X_train, Y_train)
        
    y_predicted = clf.predict(X_predict);
    return y_predicted;

def get_test_error(habitable_planets, non_habitable_planets, features, train_slice, test_slice):
    X_train, Y_train = get_X_Y(habitable_planets, non_habitable_planets, features, 0.0, train_slice);
    X_dev, Y_dev = get_X_Y(habitable_planets, non_habitable_planets, features, train_slice, test_slice);
                
    y_predicted = do_svm(X_train, Y_train, X_dev);        
    result = y_predicted * Y_dev;
        
    error = (sum(1 for i in result if i <= 0)/len(Y_dev))*100;
    
    return error;    
    
def forward_search_features (habitable, non_habitable, train_slice, dev_slice):
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
    
    return selected_features;

def load_planets_data():
    habitable_planets = np.genfromtxt('../data/habitable_planets_detailed_list.csv',filling_values = 0, names=True, dtype=None, delimiter=",",usecols=planetary_stellar_parameter_indexes);
        
    non_habitable_planets = np.genfromtxt('../data/non_habitable_planets_confirmed_detailed_list.csv', filling_values = 0, names = True, dtype=None, delimiter=",",usecols=planetary_stellar_parameter_indexes);
    
    np.random.shuffle(habitable_planets);
    np.random.shuffle(non_habitable_planets);        

    return habitable_planets, non_habitable_planets;

def find_best_features():
    try:
        print("Selecting best features");
        dictionary_of_features = dict();
        habitable_planets , non_habitable_planets = load_planets_data(); 
        for j in range(BEST_FEATURE_SELECTION_LOOP_COUNT):
            np.random.shuffle(habitable_planets);
            np.random.shuffle(non_habitable_planets);        
            selected_features = forward_search_features(habitable_planets, non_habitable_planets, TRAIN_DATA, DEV_DATA);
            frozen_selected_features = frozenset(selected_features);
            if frozen_selected_features not in dictionary_of_features:
               dictionary_of_features[frozen_selected_features] = 1;
            else:
               dictionary_of_features[frozen_selected_features] = dictionary_of_features[frozen_selected_features] + 1;
            
            print('.', end='', flush=True);
        
        for key, value in sorted(dictionary_of_features.items(), key=lambda x:x[1], reverse=True):
            print("\nBest selected features are " , key);
            return key, habitable_planets, non_habitable_planets;
    
    except ValueError:
        print('Error reading file');
        raise;

def test_features():                                       
    try:
        best_features, habitable_planets, non_habitable_planets = find_best_features();
        g_test_error = 0;
        num_of_test_iterations = 200;
        print('Testing features ', best_features, " on test data");
        for i in range(num_of_test_iterations):
            np.random.shuffle(habitable_planets);
            np.random.shuffle(non_habitable_planets);        

            #now train on larger slice (train+dev) with given feature and run test on remaining data
            train_slice = TRAIN_DATA + DEV_DATA;
            test_slice = 1.0 - train_slice;
            test_error = get_test_error(habitable_planets, non_habitable_planets, best_features, train_slice, test_slice);
        
            g_test_error = g_test_error + test_error;
            print('.', end='', flush=True);
        
        print('\nAverage test error on test data is ', g_test_error/num_of_test_iterations);
        
    
    except ValueError:
        print('Error reading file');
        raise;
        
def get_trained_model():  
     best_features, habitable_planets,non_habitable_planets  = find_best_features();

     habitable_slice_features = np.ones(habitable_planets.shape[0]);    
     non_habitable_slice_features = np.full(non_habitable_planets.shape[0], -1);
     
     habitable_slice_features = select_features(habitable_planets, habitable_slice_features, best_features);
     non_habitable_slice_features = select_features(non_habitable_planets, non_habitable_slice_features, best_features);
     
     X_train = np.vstack((habitable_slice_features[:,1:], non_habitable_slice_features[:,1:])) ;
     Y_train = np.append(habitable_slice_features[:,0], non_habitable_slice_features[:,0]);
     
     clf = svm.SVC(kernel='rbf', gamma=10)
     clf.fit(X_train, Y_train);
     
     return clf, best_features;
 
def predict_on_new_kepler_data(kepler_data_file):
    clf, features = get_trained_model();
    planets_from_kepler = np.genfromtxt(kepler_data_file, filling_values = 0, names=True, dtype=None, delimiter=",",usecols=planetary_stellar_parameter_indexes);
    
    X_data = np.ndarray(shape=(planets_from_kepler.shape[0],0));

    X_data = select_features(planets_from_kepler, X_data, features);
    
    y_predicated = clf.predict(X_data);
    
    for i in range(len(y_predicated)):
        if y_predicated[i] > 0:
            habitable_planet_koi = planets_from_kepler[i]["kepoi_name"].decode("utf-8");
            planet_temperature = planets_from_kepler[i]["koi_teq"];
            planet_radius = planets_from_kepler[i]["koi_prad"];
            print('Predicted Habitable planet koi = ',habitable_planet_koi, ", Equilibrium Surface Temperature in Celcius = ", planet_temperature - 273.15, ", Planet radius (Earth) = ", planet_radius);        

def main():
    if len(sys.argv) > 1:
        kepler_data_file = sys.argv[1];
        predict_on_new_kepler_data(kepler_data_file);
    else:
        test_features();

main()      