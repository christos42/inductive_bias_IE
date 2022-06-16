#!/bin/bash

gpu_id=0
dataset='ADE'
data_path='../data/ADE/raw'
embed_mode='albert'

cd ..
cd ..
cd ..
cd similarity_analysis

for split_id in 0 1 2 3 4 5 6 7 8 9
do
  echo 'Split: '${split_id}
  echo 'Token level'
  CUDA_VISIBLE_DEVICES=$gpu_id python similarity_analysis.py --dataset $dataset --data_path $data_path --data_name 'ade_split_'${split_id}'_test.json' --embed_mode $embed_mode --mode 'token_level' --path_trained_model 'save/e2e_training_albert_head_head_ADE_split_'${split_id}'/e2e_training_albert_head_head_ADE_split_'${split_id}'.pt'
  echo 'Aggregation: Average'
  CUDA_VISIBLE_DEVICES=$gpu_id python similarity_analysis.py --dataset $dataset --data_path $data_path --data_name 'ade_split_'${split_id}'_test.json' --embed_mode $embed_mode --mode 'word_level' --aggregation 'avg' --path_trained_model 'save/e2e_training_word_level_albert_avg_ADE_split_'${split_id}'/e2e_training_word_level_albert_avg_ADE_split_'${split_id}'.pt'
  echo 'Aggregation: Summation'
  CUDA_VISIBLE_DEVICES=$gpu_id python similarity_analysis.py --dataset $dataset --data_path $data_path --data_name 'ade_split_'${split_id}'_test.json' --embed_mode $embed_mode --mode 'word_level' --aggregation 'sum' --path_trained_model 'save/e2e_training_word_level_albert_sum_ADE_split_'${split_id}'/e2e_training_word_level_albert_sum_ADE_split_'${split_id}'.pt'
done