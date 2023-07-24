#!/bin/bash

LOG_FILE=log_1610-130723
OUT_DIR="./data"

#declare -a CLASSES=("1" "2a" "2b" "2b_swap" "2c" "3a" "3b")
declare -a CLASSES=("1")
declare -a INPUTS=("00" "01" "10" "11")

initLineNum=9
dataLines=21
dataSep=4

for cls in ${CLASSES[@]}; do
    keyword="Correlation type: $cls"
    echo $cls
    getdata="grep \"$keyword\" -A 104 $LOG_FILE"
    ln=$initLineNum
    for inp in ${INPUTS[@]};do
        echo $inp
        endln=$((ln+dataLines))
        outFile=tpr-${cls}-xy_${inp}-M_12-wtol_1e-04-ztol_1e-09.csv
        extract="$getdata | sed -n '${ln},${endln}p;$((endln+1))q' > $OUT_DIR/$outFile"
        ln=$((endln+dataSep))
        echo $extract
        eval $extract
        # Replace tab with comma
        tab2comma="sed -i 's/\t/,/g' $OUT_DIR/$outFile"
        echo $tab2comma
        eval $tab2comma
    done
done
