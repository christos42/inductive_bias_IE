#!/bin/bash

memory_alloc=64
gpu_id=0
data='ADE'
epoch=100
batch_size=20
lr=0.00002
embed_mode='biobert'
mode='e2e_training_word_level'
sub_mode='word_level'

cd ..
cd ..
echo 'Bio Clinical BERT'
echo 'Split: 0'
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:$memory_alloc CUDA_VISIBLE_DEVICES=$gpu_id python main.py --data $data --do_train --do_eval --embed_mode $embed_mode --batch_size $batch_size --lr $lr --epoch $epoch --output_file e2e_training_word_level_biobert_sum_ADE_split_0 --eval_metric macro --split_id 0 --mode $mode --sub_mode $sub_mode --word_pieces_aggregation sum
echo '--------------'
echo '--------------'
echo 'Split: 1'
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:$memory_alloc CUDA_VISIBLE_DEVICES=$gpu_id python main.py --data $data --do_train --do_eval --embed_mode $embed_mode --batch_size $batch_size --lr $lr --epoch $epoch --output_file e2e_training_word_level_biobert_sum_ADE_split_1 --eval_metric macro --split_id 1 --mode $mode --sub_mode $sub_mode --word_pieces_aggregation sum
echo '--------------'
echo '--------------'
echo 'Split: 2'
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:$memory_alloc CUDA_VISIBLE_DEVICES=$gpu_id python main.py --data $data --do_train --do_eval --embed_mode $embed_mode --batch_size $batch_size --lr $lr --epoch $epoch --output_file e2e_training_word_level_biobert_sum_ADE_split_2 --eval_metric macro --split_id 2 --mode $mode --sub_mode $sub_mode --word_pieces_aggregation sum
echo '--------------'
echo '--------------'
echo 'Split: 3'
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:$memory_alloc CUDA_VISIBLE_DEVICES=$gpu_id python main.py --data $data --do_train --do_eval --embed_mode $embed_mode --batch_size $batch_size --lr $lr --epoch $epoch --output_file e2e_training_word_level_biobert_sum_ADE_split_3 --eval_metric macro --split_id 3 --mode $mode --sub_mode $sub_mode --word_pieces_aggregation sum
echo '--------------'
echo '--------------'
echo 'Split: 4'
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:$memory_alloc CUDA_VISIBLE_DEVICES=$gpu_id python main.py --data $data --do_train --do_eval --embed_mode $embed_mode --batch_size $batch_size --lr $lr --epoch $epoch --output_file e2e_training_word_level_biobert_sum_ADE_split_4 --eval_metric macro --split_id 4 --mode $mode --sub_mode $sub_mode --word_pieces_aggregation sum
echo '--------------'
echo '--------------'
echo 'Split: 5'
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:$memory_alloc CUDA_VISIBLE_DEVICES=$gpu_id python main.py --data $data --do_train --do_eval --embed_mode $embed_mode --batch_size $batch_size --lr $lr --epoch $epoch --output_file e2e_training_word_level_biobert_sum_ADE_split_5 --eval_metric macro --split_id 5 --mode $mode --sub_mode $sub_mode --word_pieces_aggregation sum
echo '--------------'
echo '--------------'
echo 'Split: 6'
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:$memory_alloc CUDA_VISIBLE_DEVICES=$gpu_id python main.py --data $data --do_train --do_eval --embed_mode $embed_mode --batch_size $batch_size --lr $lr --epoch $epoch --output_file e2e_training_word_level_biobert_sum_ADE_split_6 --eval_metric macro --split_id 6 --mode $mode --sub_mode $sub_mode --word_pieces_aggregation sum
echo '--------------'
echo '--------------'
echo 'Split: 7'
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:$memory_alloc CUDA_VISIBLE_DEVICES=$gpu_id python main.py --data $data --do_train --do_eval --embed_mode $embed_mode --batch_size $batch_size --lr $lr --epoch $epoch --output_file e2e_training_word_level_biobert_sum_ADE_split_7 --eval_metric macro --split_id 7 --mode $mode --sub_mode $sub_mode --word_pieces_aggregation sum
echo '--------------'
echo '--------------'
echo 'Split: 8'
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:$memory_alloc CUDA_VISIBLE_DEVICES=$gpu_id python main.py --data $data --do_train --do_eval --embed_mode $embed_mode --batch_size $batch_size --lr $lr --epoch $epoch --output_file e2e_training_word_level_biobert_sum_ADE_split_8 --eval_metric macro --split_id 8 --mode $mode --sub_mode $sub_mode --word_pieces_aggregation sum
echo '--------------'
echo '--------------'
echo 'Split: 9'
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:$memory_alloc CUDA_VISIBLE_DEVICES=$gpu_id python main.py --data $data --do_train --do_eval --embed_mode $embed_mode --batch_size $batch_size --lr $lr --epoch $epoch --output_file e2e_training_word_level_biobert_sum_ADE_split_9 --eval_metric macro --split_id 9 --mode $mode --sub_mode $sub_mode --word_pieces_aggregation sum
echo '###################################################################################################################'