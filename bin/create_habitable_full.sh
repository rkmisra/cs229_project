#!/bin/sh
awk '{print;}' ../data/habitable_planets_koi.csv | xargs -n 1 -I % grep % ../data/cumulative.csv > ../data/habitable_planets_detailed_list.csv
