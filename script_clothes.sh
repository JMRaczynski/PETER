#!/bin/bash

for i in {1..5}
do
  python -u main.py --data_path ../datasets/Amazon/ClothingShoesAndJewelry/reviews.pickle \
  --index_dir ../datasets/Amazon/ClothingShoesAndJewelry/$i --outf generated_clothes.txt \
  --checkpoint ./models/AmazonClothes/ --use_feature --cuda --peter_mask --epochs 1 | tee script_output_clothes_peter_$i.log
done