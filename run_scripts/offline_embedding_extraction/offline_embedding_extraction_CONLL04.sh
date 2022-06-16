#!/bin/bash

gpu_id=0
dataset='CONLL04'
path_out='offline_embeddings/'

cd ..
cd ..
# Word Level: BERT
echo 'Word Level: BERT'
echo 'Train set'
CUDA_VISIBLE_DEVICES=$gpu_id python offline_embedding_extraction.py --dataset $dataset --embed_mode bert_cased --set train --split_id -1 --mode word_level --word_pieces_aggregation avg --path_out $path_out
CUDA_VISIBLE_DEVICES=$gpu_id python offline_embedding_extraction.py --dataset $dataset --embed_mode bert_cased --set train --split_id -1 --mode word_level --word_pieces_aggregation sum --path_out $path_out
echo '__________'
echo 'Development set'
CUDA_VISIBLE_DEVICES=$gpu_id python offline_embedding_extraction.py --dataset $dataset --embed_mode bert_cased --set dev --split_id -1 --mode word_level --word_pieces_aggregation avg --path_out $path_out
CUDA_VISIBLE_DEVICES=$gpu_id python offline_embedding_extraction.py --dataset $dataset --embed_mode bert_cased --set dev --split_id -1 --mode word_level --word_pieces_aggregation sum --path_out $path_out
echo '__________'
echo 'Test set'
CUDA_VISIBLE_DEVICES=$gpu_id python offline_embedding_extraction.py --dataset $dataset --embed_mode bert_cased --set test --split_id -1 --mode word_level --word_pieces_aggregation avg --path_out $path_out
CUDA_VISIBLE_DEVICES=$gpu_id python offline_embedding_extraction.py --dataset $dataset --embed_mode bert_cased --set test --split_id -1 --mode word_level --word_pieces_aggregation sum --path_out $path_out
echo '#################################################################################################################'
# Word Level: ALBERT
echo 'Word Level: ALBERT'
echo 'Train set'
CUDA_VISIBLE_DEVICES=$gpu_id python offline_embedding_extraction.py --dataset $dataset --embed_mode albert --set train --split_id -1 --mode word_level --word_pieces_aggregation avg --path_out $path_out
CUDA_VISIBLE_DEVICES=$gpu_id python offline_embedding_extraction.py --dataset $dataset --embed_mode albert --set train --split_id -1 --mode word_level --word_pieces_aggregation sum --path_out $path_out
echo '__________'
echo 'Development set'
CUDA_VISIBLE_DEVICES=$gpu_id python offline_embedding_extraction.py --dataset $dataset --embed_mode albert --set dev --split_id -1 --mode word_level --word_pieces_aggregation avg --path_out $path_out
CUDA_VISIBLE_DEVICES=$gpu_id python offline_embedding_extraction.py --dataset $dataset --embed_mode albert --set dev --split_id -1 --mode word_level --word_pieces_aggregation sum --path_out $path_out
echo '__________'
echo 'Test set'
CUDA_VISIBLE_DEVICES=$gpu_id python offline_embedding_extraction.py --dataset $dataset --embed_mode albert --set test --split_id -1 --mode word_level --word_pieces_aggregation avg --path_out $path_out
CUDA_VISIBLE_DEVICES=$gpu_id python offline_embedding_extraction.py --dataset $dataset --embed_mode albert --set test --split_id -1 --mode word_level --word_pieces_aggregation sum --path_out $path_out
echo '#################################################################################################################'
# Word Level: CharacterBERT (general)
echo 'Word Level: CharacterBERT'
echo 'Train set'
CUDA_VISIBLE_DEVICES=$gpu_id python offline_embedding_extraction.py --dataset $dataset --embed_mode characterBERT --set train --split_id -1 --mode word_level --path_out $path_out
echo '__________'
echo 'Development set'
CUDA_VISIBLE_DEVICES=$gpu_id python offline_embedding_extraction.py --dataset $dataset --embed_mode characterBERT --set dev --split_id -1 --mode word_level --path_out $path_out
echo '__________'
echo 'Test set'
CUDA_VISIBLE_DEVICES=$gpu_id python offline_embedding_extraction.py --dataset $dataset --embed_mode characterBERT --set test --split_id -1 --mode word_level --path_out $path_out
echo '#################################################################################################################'
# Word Level: CANINE-c
echo 'Word Level: CANINE-c'
echo 'Train set'
CUDA_VISIBLE_DEVICES=$gpu_id python offline_embedding_extraction.py --dataset $dataset --embed_mode canine_c --set train --split_id -1 --mode word_level --word_pieces_aggregation avg --path_out $path_out
CUDA_VISIBLE_DEVICES=$gpu_id python offline_embedding_extraction.py --dataset $dataset --embed_mode canine_c --set train --split_id -1 --mode word_level --word_pieces_aggregation sum --path_out $path_out
echo '__________'
echo 'Development set'
CUDA_VISIBLE_DEVICES=$gpu_id python offline_embedding_extraction.py --dataset $dataset --embed_mode canine_c --set dev --split_id -1 --mode word_level --word_pieces_aggregation avg --path_out $path_out
CUDA_VISIBLE_DEVICES=$gpu_id python offline_embedding_extraction.py --dataset $dataset --embed_mode canine_c --set dev --split_id -1 --mode word_level --word_pieces_aggregation sum --path_out $path_out
echo '__________'
echo 'Test set'
CUDA_VISIBLE_DEVICES=$gpu_id python offline_embedding_extraction.py --dataset $dataset --embed_mode canine_c --set test --split_id -1 --mode word_level --word_pieces_aggregation avg --path_out $path_out
CUDA_VISIBLE_DEVICES=$gpu_id python offline_embedding_extraction.py --dataset $dataset --embed_mode canine_c --set test --split_id -1 --mode word_level --word_pieces_aggregation sum --path_out $path_out
echo '#################################################################################################################'
# Word Level: CANINE-s
echo 'Word Level: CANINE-s'
echo 'Train set'
CUDA_VISIBLE_DEVICES=$gpu_id python offline_embedding_extraction.py --dataset $dataset --embed_mode canine_s --set train --split_id -1 --mode word_level --word_pieces_aggregation avg --path_out $path_out
CUDA_VISIBLE_DEVICES=$gpu_id python offline_embedding_extraction.py --dataset $dataset --embed_mode canine_s --set train --split_id -1 --mode word_level --word_pieces_aggregation sum --path_out $path_out
echo '__________'
echo 'Development set'
CUDA_VISIBLE_DEVICES=$gpu_id python offline_embedding_extraction.py --dataset $dataset --embed_mode canine_s --set dev --split_id -1 --mode word_level --word_pieces_aggregation avg --path_out $path_out
CUDA_VISIBLE_DEVICES=$gpu_id python offline_embedding_extraction.py --dataset $dataset --embed_mode canine_s --set dev --split_id -1 --mode word_level --word_pieces_aggregation sum --path_out $path_out
echo '__________'
echo 'Test set'
CUDA_VISIBLE_DEVICES=$gpu_id python offline_embedding_extraction.py --dataset $dataset --embed_mode canine_s --set test --split_id -1 --mode word_level --word_pieces_aggregation avg --path_out $path_out
CUDA_VISIBLE_DEVICES=$gpu_id python offline_embedding_extraction.py --dataset $dataset --embed_mode canine_s --set test --split_id -1 --mode word_level --word_pieces_aggregation sum --path_out $path_out
echo '#################################################################################################################'
#################################################################################################################################################################################