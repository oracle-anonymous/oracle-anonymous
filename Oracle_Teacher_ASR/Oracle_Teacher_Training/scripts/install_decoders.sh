#!/bin/sh
set -xe

git clone https://github.com/PaddlePaddle/DeepSpeech
cd DeepSpeech
git checkout a76fc69
cd ..
mv DeepSpeech/decoders/swig_wrapper.py DeepSpeech/decoders/swig/ctc_decoders.py
mv DeepSpeech/decoders/swig ./decoders
rm -rf DeepSpeech

cd decoders
chmod +x setup.sh
./setup.sh
cd ..
