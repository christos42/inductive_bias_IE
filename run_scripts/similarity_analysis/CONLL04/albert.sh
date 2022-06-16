#!/bin/bash

gpu_id=0
dataset='CONLL04'
data_path='../data/CONLL04/'
embed_mode='albert'

cd ..
cd ..
cd ..
cd similarity_analysis

for run_id in 1 2 3 4 5
do
  echo 'Run: '${run_id}
  echo 'Token level'
  CUDA_VISIBLE_DEVICES=$gpu_id python similarity_analysis.py --dataset $dataset --data_path $data_path --data_name 'test_triples.json' --embed_mode $embed_mode --mode 'token_level' --path_trained_model 'saved_models/CONLL04/run_'${run_id}'/macro/e2e_training/e2e_training_albert_head_head_CONLL04_macro/e2e_training_albert_head_head_CONLL04_macro.pt'
  CUDA_VISIBLE_DEVICES=$gpu_id python similarity_analysis.py --dataset $dataset --data_path $data_path --data_name 'test_triples.json' --embed_mode $embed_mode --mode 'token_level' --path_trained_model 'saved_models/CONLL04/run_'${run_id}'/micro/e2e_training/e2e_training_albert_head_head_CONLL04_micro/e2e_training_albert_head_head_CONLL04_micro.pt'
  echo 'Aggregation: Average'
  CUDA_VISIBLE_DEVICES=$gpu_id python similarity_analysis.py --dataset $dataset --data_path $data_path --data_name 'test_triples.json' --embed_mode $embed_mode --mode 'word_level' --aggregation 'avg' --path_trained_model 'saved_models/CONLL04/run_'${run_id}'/macro/e2e_training_word_level/e2e_training_word_level_albert_avg_CONLL04_macro/e2e_training_word_level_albert_avg_CONLL04_macro.pt'
  CUDA_VISIBLE_DEVICES=$gpu_id python similarity_analysis.py --dataset $dataset --data_path $data_path --data_name 'test_triples.json' --embed_mode $embed_mode --mode 'word_level' --aggregation 'avg' --path_trained_model 'saved_models/CONLL04/run_'${run_id}'/micro/e2e_training_word_level/e2e_training_word_level_albert_avg_CONLL04_micro/e2e_training_word_level_albert_avg_CONLL04_micro.pt'
  echo 'Aggregation: Summation'
  CUDA_VISIBLE_DEVICES=$gpu_id python similarity_analysis.py --dataset $dataset --data_path $data_path --data_name 'test_triples.json' --embed_mode $embed_mode --mode 'word_level' --aggregation 'sum' --path_trained_model 'saved_models/CONLL04/run_'${run_id}'/macro/e2e_training_word_level/e2e_training_word_level_albert_sum_CONLL04_macro/e2e_training_word_level_albert_sum_CONLL04_macro.pt'
  CUDA_VISIBLE_DEVICES=$gpu_id python similarity_analysis.py --dataset $dataset --data_path $data_path --data_name 'test_triples.json' --embed_mode $embed_mode --mode 'word_level' --aggregation 'sum' --path_trained_model 'saved_models/CONLL04/run_'${run_id}'/micro/e2e_training_word_level/e2e_training_word_level_albert_sum_CONLL04_micro/e2e_training_word_level_albert_sum_CONLL04_micro.pt'
done
