#!/bin/sh
rm -rf ../data/non_habitable_planets_detailed_list.csv
while read line
do
	koi=`echo $line | cut -d"," -f3`
	grep "$koi" ../data/habitable_planets_detailed_list.csv
	if [ $? -ne 0 ]
	then
		echo $line  >> ../data/non_habitable_planets_detailed_list.csv
	fi
done < ../data/cumulative.csv 
