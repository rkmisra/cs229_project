#!/bin/sh
cat $1 | awk -F "," '{printf("%s\n",$1);}'
