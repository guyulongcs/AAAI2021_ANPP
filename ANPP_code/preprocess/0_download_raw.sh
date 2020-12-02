#! /bin/bash


folder=$1
file_review=$2
file_meta=$3

file_review_gz="${file_review}.gz"
file_meta_gz="${file_meta}.gz"

mkdir -p ${folder}
cd ${folder}


#wget -c http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/${file_review_gz}
gzip -d ${file_review_gz}

#wget -c http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/${file_meta_gz}
gzip -d ${file_meta_gz}

