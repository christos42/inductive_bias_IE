#!/bin/bash

gpu_id=0
dataset='ADE'
path_out='offline_embeddings/'

cd ..
cd ..
# Word Level: BERT
echo 'Word Level: BERT'
echo 'Train set'
for split_id in 0 1 2 3 4 5 6 7 8 9
do
  echo 'Split:' $split_id
  CUDA_VISIBLE_DEVICES=$gpu_id python offline_embedding_extraction.py --dataset $dataset --embed_mode bert_cased --set train --split_id $split_id --mode word_level --word_pieces_aggregation avg --path_out $path_out
  CUDA_VISIBLE_DEVICES=$gpu_id python offline_embedding_extraction.py --dataset $dataset --embed_mode bert_cased --set train --split_id $split_id --mode word_level --word_pieces_aggregation sum --path_out $path_out
done
echo '__________'
echo 'Test set'
for split_id in 0 1 2 3 4 5 6 7 8 9
do
  echo 'Split:' $split_id
  CUDA_VISIBLE_DEVICES=$gpu_id python offline_embedding_extraction.py --dataset $dataset --embed_mode bert_cased --set test --split_id $split_id --mode word_level --word_pieces_aggregation avg --path_out $path_out
  CUDA_VISIBLE_DEVICES=$gpu_id python offline_embedding_extraction.py --dataset $dataset --embed_mode bert_cased --set test --split_id $split_id --mode word_level --word_pieces_aggregation sum --path_out $path_out
done
echo '#################################################################################################################'
# Word Level: ALBERT
echo 'Word Level: ALBERT'
echo 'Train set'
for split_id in 0 1 2 3 4 5 6 7 8 9
do
  echo 'Split:' $split_id
  CUDA_VISIBLE_DEVICES=$gpu_id python offline_embedding_extraction.py --dataset $dataset --embed_mode albert --set train --split_id $split_id --mode word_level --word_pieces_aggregation avg --path_out $path_out
  CUDA_VISIBLE_DEVICES=$gpu_id python offline_embedding_extraction.py --dataset $dataset --embed_mode albert --set train --split_id $split_id --mode word_level --word_pieces_aggregation sum --path_out $path_out
done
echo '__________'
echo 'Test set'
for split_id in 0 1 2 3 4 5 6 7 8 9
do
  echo 'Split:' $split_id
  CUDA_VISIBLE_DEVICES=$gpu_id python offline_embedding_extraction.py --dataset $dataset --embed_mode albert --set test --split_id $split_id --mode word_level --word_pieces_aggregation avg --path_out $path_out
  CUDA_VISIBLE_DEVICES=$gpu_id python offline_embedding_extraction.py --dataset $dataset --embed_mode albert --set test --split_id $split_id --mode word_level --word_pieces_aggregation sum --path_out $path_out
done
echo '#################################################################################################################'
# Word Level: Bio Clinical BERT
echo 'Word Level: Bio Clinical BERT'
echo 'Train set'
for split_id in 0 1 2 3 4 5 6 7 8 9
do
  echo 'Split:' $split_id
  CUDA_VISIBLE_DEVICES=$gpu_id python offline_embedding_extraction.py --dataset $dataset --embed_mode biobert --set train --split_id $split_id --mode word_level --word_pieces_aggregation avg --path_out $path_out
  CUDA_VISIBLE_DEVICES=$gpu_id python offline_embedding_extraction.py --dataset $dataset --embed_mode biobert --set train --split_id $split_id --mode word_level --word_pieces_aggregation sum --path_out $path_out
done
echo '__________'
echo 'Test set'
for split_id in 0 1 2 3 4 5 6 7 8 9
do
  echo 'Split:' $split_id
  CUDA_VISIBLE_DEVICES=$gpu_id python offline_embedding_extraction.py --dataset $dataset --embed_mode biobert --set test --split_id $split_id --mode word_level --word_pieces_aggregation avg --path_out $path_out
  CUDA_VISIBLE_DEVICES=$gpu_id python offline_embedding_extraction.py --dataset $dataset --embed_mode biobert --set test --split_id $split_id --mode word_level --word_pieces_aggregation sum --path_out $path_out
done
echo '#################################################################################################################'
# Word Level: CharacterBERT (general)
echo 'Word Level: CharacterBERT'
echo 'Train set'
for split_id in 0 1 2 3 4 5 6 7 8 9
do
  echo 'Split:' $split_id
  CUDA_VISIBLE_DEVICES=$gpu_id python offline_embedding_extraction.py --dataset $dataset --embed_mode characterBERT --set train --split_id $split_id --mode word_level --path_out $path_out
done
echo '__________'
echo 'Test set'
for split_id in 0 1 2 3 4 5 6 7 8 9
do
  echo 'Split:' $split_id
  CUDA_VISIBLE_DEVICES=$gpu_id python offline_embedding_extraction.py --dataset $dataset --embed_mode characterBERT --set test --split_id $split_id --mode word_level --path_out $path_out
done
echo '#################################################################################################################'
# Word Level: CANINE-c
echo 'Word Level: CANINE-c'
echo 'Train set'
for split_id in 0 1 2 3 4 5 6 7 8 9
do
  echo 'Split:' $split_id
  CUDA_VISIBLE_DEVICES=$gpu_id python offline_embedding_extraction.py --dataset $dataset --embed_mode canine_c --set train --split_id $split_id --mode word_level --word_pieces_aggregation avg --path_out $path_out
  CUDA_VISIBLE_DEVICES=$gpu_id python offline_embedding_extraction.py --dataset $dataset --embed_mode canine_c --set train --split_id $split_id --mode word_level --word_pieces_aggregation sum --path_out $path_out
done
echo '__________'
echo 'Test set'
for split_id in 0 1 2 3 4 5 6 7 8 9
do
  echo 'Split:' $split_id
  CUDA_VISIBLE_DEVICES=$gpu_id python offline_embedding_extraction.py --dataset $dataset --embed_mode canine_c --set test --split_id $split_id --mode word_level --word_pieces_aggregation avg --path_out $path_out
  CUDA_VISIBLE_DEVICES=$gpu_id python offline_embedding_extraction.py --dataset $dataset --embed_mode canine_c --set test --split_id $split_id --mode word_level --word_pieces_aggregation sum --path_out $path_out
done
echo '#################################################################################################################'
# Word Level: CANINE-c
echo 'Word Level: CANINE-s'
echo 'Train set'
for split_id in 0 1 2 3 4 5 6 7 8 9
do
  echo 'Split:' $split_id
  CUDA_VISIBLE_DEVICES=$gpu_id python offline_embedding_extraction.py --dataset $dataset --embed_mode canine_s --set train --split_id $split_id --mode word_level --word_pieces_aggregation avg --path_out $path_out
  CUDA_VISIBLE_DEVICES=$gpu_id python offline_embedding_extraction.py --dataset $dataset --embed_mode canine_s --set train --split_id $split_id --mode word_level --word_pieces_aggregation sum --path_out $path_out
done
echo '__________'
echo 'Test set'
for split_id in 0 1 2 3 4 5 6 7 8 9
do
  echo 'Split:' $split_id
  CUDA_VISIBLE_DEVICES=$gpu_id python offline_embedding_extraction.py --dataset $dataset --embed_mode canine_s --set test --split_id $split_id --mode word_level --word_pieces_aggregation avg --path_out $path_out
  CUDA_VISIBLE_DEVICES=$gpu_id python offline_embedding_extraction.py --dataset $dataset --embed_mode canine_s --set test --split_id $split_id --mode word_level --word_pieces_aggregation sum --path_out $path_out
done
#################################################################################################################################################################################