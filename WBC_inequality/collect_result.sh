#!/bin/bash

LOGFILE=log_2352-230823
OUTFILE=asymp_br_wbc.csv
HEADER='# delta, entropy, win_prob, lambda, C_lambda'
declare -a KeyWords=('delta=' 'Entropy:' 'WinProb:' 'Lambda:' 'C_lambda:')
declare -a Delimiters=('=' ':' ':' ':' ':')

# Write HEADER to OUTFILE
echo "$HEADER" >> $OUTFILE

num_params=${#KeyWords[@]}  # Num of params
IFS=','                     # Internal file separator; use for conacating array elements
COUNTER=1                   # Count the number of sets of data
BREAK=false                 # Break flag

# Collect data from log file and append to OUTFILE
while true;
do
    declare -a data_arr=()
    for (( i=0; i<num_params; i++ ));
    do
        key_word=${KeyWords[i]}
        delimiter=${Delimiters[i]}
        data=$(grep "$key_word" $LOGFILE | cut -d "$delimiter" -f 2 | cut -d $'\n' -f $COUNTER)
        if [[ -z $data ]];then
            BREAK=true
            break;
        fi
        data_arr+=("${data}")
    done
    if [ "$BREAK" = true ];then break; fi;
    echo "${data_arr[*]}" >> $OUTFILE
    (( COUNTER++ ))
done
