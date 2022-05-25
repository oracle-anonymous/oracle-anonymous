#!/bin/bash

if [ $# -ne 2 ]; then
        echo "Specifiy csv (such as librivox-train-all.csv)"
        exit 1;
fi

cut -d "," -f1 $1 > at_file
sed -i 's#/data/librispeech/LibriSpeech/#/knowledge_of_the_oracle_teacher/#g' at_file
sed -i 's/\.wav/\.new/' at_file
sed -i 's#/home#,/home#g' at_file
sed -i 's#wav_filename#,teacher_kd#g' at_file

paste -d "\0" $1 at_file > $2 


rm -rf at_file
