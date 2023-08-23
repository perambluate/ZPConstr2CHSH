#!/bin/bash

DIR=./

print_usage() {
  printf "Usage: i) bash run_bg.sh -t TYPE -m METHOD\n"
  printf "      ii) bash run_bg.sh -d DIR -f FILE\n"
  printf "TYPE          : blind|one|two.\n"
  printf "METHOD [BFF21]: BFF21(bff21)|BBS20(bbs20)|FM18(fm18) (Specify when TYPE=blind)\n"
  printf "DIR           : Directory to run the file.\n"
  printf "FILE          : Python file to run.\n"
}

while getopts t:m:d:f: flag
do
    case "${flag}" in
        t) TYPE=${OPTARG};;
        m) METHOD=${OPTARG};;
        d) DIR=${OPTARG};;
        f) FILE=${OPTARG};;
        *) print_usage
            exit 1 ;;
    esac
done

if ! [ -n "$FILE" ]; then
    case $TYPE in
        blind)
            DIR=./blindRandomness
            if [ -n "$METHOD" ];then
                case $METHOD in
                    BFF21|bff21) FILE=bff21/blindRandomness-BFF21.py;;
                    BBS20|bbs20) FILE=bbs20/blindRandomness-BBS20.py;;
                    FM18|fm18) FILE=fm18/blindRandomness-FM18.py;;
                    *) echo "Wrong METHOD!"
                        exit 1 ;;
                esac
            else
                FILE=bff21/blindRandomness-BFF21.py
            fi
            ;;
        one|two)
            DIR=./${TYPE}PartyRandomness
            FILE=${TYPE}PartyRandomness.py
            ;;
        *)  echo "Neither TYPE is wrong nor \
                    neither TYPE nor (DIR, FILE) not specify"
            print_usage
            exit 1 ;;
    esac
fi

echo "python ${DIR}/${FILE}"

# Save log file as backup or for debugging
LOG=log_$(date +%H%M-%d%m%y)

# Run in background
cd $DIR && \
nohup python $FILE > $LOG 2>&1 &
