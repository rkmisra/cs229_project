#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ramisra
"""

import sys
import numpy as np
from sklearn import svm  
import matplotlib.pyplot as plt       

TRAIN_DATA = 0.5;
DEV_DATA = 0.2;
BEST_FEATURE_SELECTION_LOOP_COUNT=50

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
planetary_stellar_parameter_cols_dict = {   "koi_period":   "Orbital Period",
                                       "koi_ror":     "Planet-Star Radius Ratio",
                                       "koi_srho":      "Fitted Stellar Density",
                                       "koi_prad":     "Planetary Radius",
                                       "koi_sma":      "Orbit Semi-Major Axis",
                                       "koi_teq":       "Equilibrium Temperature",
                                       "koi_insol":     "Insolation Flux",
                                       "koi_dor":       "Planet-Star Distance over Star Radius",
                                       "koi_count":     "Number of Planet" ,
                                       "koi_steff":     "Stellar Effective Temperature" ,
                                       "koi_slogg":     "Stellar Surface Gravity",
                                       "koi_smet":      "Stellar Metallicity",
                                       "koi_srad":      "Stellar Radius",
                                       "koi_smass":      "Stellar Mass"
                                       };
                                
                                

def get_svm():
    return svm.SVC(kernel='rbf', gamma=10, class_weight = 'balanced')

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
    clf = get_svm()
    clf.fit(X_train, Y_train)
        
    y_predicted = clf.predict(X_predict);
    return y_predicted;

def get_test_error(habitable_planets, non_habitable_planets, features, start_train, train_width, start_test, test_width):
    X_train, Y_train = get_X_Y(habitable_planets, non_habitable_planets, features, start_train, train_width);
    X_dev, Y_dev = get_X_Y(habitable_planets, non_habitable_planets, features, start_test, test_width);
                
    y_predicted = do_svm(X_train, Y_train, X_dev);        
    result = y_predicted * Y_dev;
        
    error = (sum(1 for i in result if i <= 0)/len(Y_dev))*100;
    
    return error;    
    
def forward_search_features (habitable, non_habitable, start_train, train_width, start_test, test_width):
    selected_features = set([]);
    previous_min_error = 0;
    for i in planetary_stellar_parameter_cols:
        min_error = 100;        
        min_index = 0;
        for j in planetary_stellar_parameter_cols:
            if j not in selected_features:
                tmp_selected_features = set(selected_features);
                tmp_selected_features.add(j);
                
                error = get_test_error(habitable, non_habitable, tmp_selected_features, start_train, train_width, start_test, test_width);
                
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
    
    np.random.shuffle(habitable_planets)
    np.random.shuffle(non_habitable_planets)        

    return habitable_planets, non_habitable_planets;

def find_best_features():
    try:
        print("Selecting best features");
        dictionary_of_features = dict();
        habitable_planets , non_habitable_planets = load_planets_data(); 
        for j in range(BEST_FEATURE_SELECTION_LOOP_COUNT):
            habitable_size = len(habitable_planets);
            non_habitable_size = len(non_habitable_planets);

            #only resuffule train/dev set  
            np.random.shuffle(habitable_planets[0: int(habitable_size * (TRAIN_DATA + DEV_DATA))])
            np.random.shuffle(non_habitable_planets[0:int(non_habitable_size * (TRAIN_DATA + DEV_DATA))])        
            
            selected_features = forward_search_features(habitable_planets, non_habitable_planets, 0.0, TRAIN_DATA, TRAIN_DATA, DEV_DATA);
            frozen_selected_features = frozenset(selected_features);
            if frozen_selected_features not in dictionary_of_features:
               dictionary_of_features[frozen_selected_features] = 1;
            else:
               dictionary_of_features[frozen_selected_features] = dictionary_of_features[frozen_selected_features] + 1;
            
            print('.', end='', flush=True);

        # select top 4 set
        TOP_NUMBER_OF_FEATURES = 5
        index = 0;
        high_frequency_found = False;
        min_dev_error = 0;
        best_feature_set = []
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for key, value in sorted(dictionary_of_features.items(), key=lambda x:x[1], reverse=True):
            if index == TOP_NUMBER_OF_FEATURES:
                break;
            index +=1
            
            dev_error = get_test_error(habitable_planets, non_habitable_planets, key, 0.0, TRAIN_DATA, TRAIN_DATA, DEV_DATA)
            print("\nFeature set = ", key , " number of times selected = ", value, " and dev set error ", dev_error);
            feature_label = []
            for feature in key:
                feature_label.append(planetary_stellar_parameter_cols_dict[feature])
            ax.scatter(dev_error, value, label=feature_label)
            # best feature is which is selected at least 15% of time of BEST_FEATURE_SELECTION_LOOP_COUNT (default = 30)
            # and with lowest dev error, otherwise simply lowest dev_error 
            if value >= BEST_FEATURE_SELECTION_LOOP_COUNT * 0.15:
                if high_frequency_found == False:
                        best_feature_set = key
                        min_dev_error = dev_error
                        high_frequency_found = True
                elif dev_error < min_dev_error:
                        best_feature_set = key
                        min_dev_error = dev_error
            elif high_frequency_found == False:
                    if dev_error < min_dev_error:
                        best_feature_set = key
                        min_dev_error = dev_error
        
        Y_annotate = dictionary_of_features[best_feature_set]
        X_annotate = min_dev_error
        
        ax.set_xlim([0, 20])
        ax.set_ylim([0, 30])
        best_feature_label = ""
        for feature in best_feature_set:
                best_feature_label += " {" + planetary_stellar_parameter_cols_dict[feature] + "} "
                
        ax.annotate('Feature selected = ' + best_feature_label, xy = (X_annotate, Y_annotate), xytext = (X_annotate + 2, Y_annotate + 2), 
                     arrowprops=dict(facecolor='black', shrink=0.05),)
           
        ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           mode="expand", borderaxespad=0.)
        plt.xlabel('% Dev set error with selected feature')
        plt.ylabel('Number of times feature selected')
        plt.show()
        
        print("\nBest selected features are " , best_feature_set)
        return best_feature_set, habitable_planets, non_habitable_planets;
    
    except ValueError:
        print('Error reading file');
        raise;

def test_features():                                       
    try:
        best_features, habitable_planets, non_habitable_planets = find_best_features();
        
        #error on test data
        test_error = get_test_error(habitable_planets, non_habitable_planets, best_features, 0.0, TRAIN_DATA, TRAIN_DATA + DEV_DATA, 1.0);
        print('\ntest error on test data is ', test_error);
        
        # error on train data
        train_error = get_test_error(habitable_planets, non_habitable_planets, best_features, 0.0, TRAIN_DATA, 0.0, TRAIN_DATA);
        print('Error on training data is ', train_error);

        # error on dev data
        dev_error = get_test_error(habitable_planets, non_habitable_planets, best_features,  0.0, TRAIN_DATA, TRAIN_DATA, DEV_DATA);
        print('Error on dev data is ', dev_error);
         
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
     
     clf = get_svm()
     clf.fit(X_train, Y_train);
     
     return clf, best_features;
 
def predict_on_new_kepler_data(kepler_data_file):
    clf, features = get_trained_model();
    planets_from_kepler = np.genfromtxt(kepler_data_file, filling_values = 0, names=True, dtype=None, delimiter=",",usecols=planetary_stellar_parameter_indexes);
    
    X_data = np.ndarray(shape=(planets_from_kepler.shape[0],0));

    X_data = select_features(planets_from_kepler, X_data, features);
    
    y_predicated = clf.predict(X_data);
    
    X_distance_from_parent_star = []
    Y_surface_temprature = []
    S_planet_radius = []
    colors = []
    color_space = 255*255*255
    
    for i in range(len(y_predicated)):
        if y_predicated[i] > 0:
            habitable_planet_koi = planets_from_kepler[i]["kepoi_name"].decode("utf-8");
            planet_temperature = planets_from_kepler[i]["koi_teq"] - 273.15;
            planet_radius = planets_from_kepler[i]["koi_prad"];
            planet_star_distance = planets_from_kepler[i]["koi_dor"]
            print('Predicted Habitable planet koi = ',habitable_planet_koi, ", Equilibrium Surface Temperature in Celcius = ", planet_temperature, ", Planet radius (Earth) = ", planet_radius);        
            X_distance_from_parent_star.append(planet_star_distance)
            Y_surface_temprature.append(planet_temperature)
            S_planet_radius.append(planet_radius)
            colors.append(np.random.randint(color_space))
    
    plt.scatter(X_distance_from_parent_star, Y_surface_temprature, s = S_planet_radius, c = colors)
    plt.xlabel('Distance from parent star')
    plt.ylabel('Planet surface temprature in celcius')
    plt.show()

'''
This program could be called either with no argument
or data file with all columns generated from Kepler's KOI table
 
https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative

The first line of data file needs to be column names.

One example of how to run this is

python predict_habitability.py  ../data/cumulative_test.csv

In this case, it prints KOI of all planets which it has
identified as potentially habitable.

If this is called without any argument, it just finds best feature from
training/dev data and reports the error on training and test data.
'''
def main():
    if len(sys.argv) > 1:
        kepler_data_file = sys.argv[1];
        predict_on_new_kepler_data(kepler_data_file);
    else:
        test_features();

main()      