#!/usr/bin/env bash
wget http://mi.soi.city.ac.uk/datasets/magnatagatune/mp3.zip.001
wget http://mi.soi.city.ac.uk/datasets/magnatagatune/mp3.zip.002
wget http://mi.soi.city.ac.uk/datasets/magnatagatune/mp3.zip.003
mkdir -p ~/MagnaTagATune/mp3
cat mp3.zip.* > single.zip
unzip single.zip -d MagnaTagATune/mp3/
cd ~/MagnaTagATune
wget http://mi.soi.city.ac.uk/datasets/magnatagatune/comparisons_final.csv
wget http://mi.soi.city.ac.uk/datasets/magnatagatune/clip_info_final.csv