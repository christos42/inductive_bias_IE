#!/bin/bash

memory_alloc=32
gpu_id=0
data='CONLL04'
epoch=200
batch_size=10
lr=0.00002
clip=1
mode='e2e_training_word_level'
sub_mode='word_level'

cd ..
cd ..
echo 'BERT cased'
embed_mode='bert_cased'
echo 'Macro'
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:$memory_alloc CUDA_VISIBLE_DEVICES=$gpu_id python main.py --data $data --do_train --do_eval --embed_mode $embed_mode --batch_size $batch_size --lr $lr --epoch $epoch --seed 42 --output_file e2e_training_word_level_bert_avg_CONLL04_macro --eval_metric macro --mode $mode --sub_mode $sub_mode --word_pieces_aggregation avg
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:$memory_alloc CUDA_VISIBLE_DEVICES=$gpu_id python main.py --data $data --do_train --do_eval --embed_mode $embed_mode --batch_size $batch_size --lr $lr --epoch $epoch --seed 42 --output_file e2e_training_word_level_bert_sum_CONLL04_macro --eval_metric macro --mode $mode --sub_mode $sub_mode --word_pieces_aggregation sum
echo 'Micro'
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:$memory_alloc CUDA_VISIBLE_DEVICES=$gpu_id python main.py --data $data --do_train --do_eval --embed_mode $embed_mode --batch_size $batch_size --lr $lr --epoch $epoch --seed 42 --output_file e2e_training_word_level_bert_avg_CONLL04_micro --eval_metric micro --mode $mode --sub_mode $sub_mode --word_pieces_aggregation avg
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:$memory_alloc CUDA_VISIBLE_DEVICES=$gpu_id python main.py --data $data --do_train --do_eval --embed_mode $embed_mode --batch_size $batch_size --lr $lr --epoch $epoch --seed 42 --output_file e2e_training_word_level_bert_sum_CONLL04_micro --eval_metric micro --mode $mode --sub_mode $sub_mode --word_pieces_aggregation sum
echo '--------------'
echo '--------------'
echo 'ALBERT'
embed_mode='albert'
echo 'Macro'
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:$memory_alloc CUDA_VISIBLE_DEVICES=$gpu_id python main.py --data $data --do_train --do_eval --embed_mode $embed_mode --batch_size $batch_size --lr $lr --epoch $epoch --seed 42 --output_file e2e_training_word_level_albert_avg_CONLL04_macro --eval_metric macro --mode $mode --sub_mode $sub_mode --word_pieces_aggregation avg
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:$memory_alloc CUDA_VISIBLE_DEVICES=$gpu_id python main.py --data $data --do_train --do_eval --embed_mode $embed_mode --batch_size $batch_size --lr $lr --epoch $epoch --seed 42 --output_file e2e_training_word_level_albert_sum_CONLL04_macro --eval_metric macro --mode $mode --sub_mode $sub_mode --word_pieces_aggregation sum
echo 'Micro'
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:$memory_alloc CUDA_VISIBLE_DEVICES=$gpu_id python main.py --data $data --do_train --do_eval --embed_mode $embed_mode --batch_size $batch_size --lr $lr --epoch $epoch --seed 42 --output_file e2e_training_word_level_albert_avg_CONLL04_micro --eval_metric micro --mode $mode --sub_mode $sub_mode --word_pieces_aggregation avg
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:$memory_alloc CUDA_VISIBLE_DEVICES=$gpu_id python main.py --data $data --do_train --do_eval --embed_mode $embed_mode --batch_size $batch_size --lr $lr --epoch $epoch --seed 42 --output_file e2e_training_word_level_albert_sum_CONLL04_micro --eval_metric micro --mode $mode --sub_mode $sub_mode --word_pieces_aggregation sum

echo '--------------'
echo '--------------'
echo 'CANINE-C'
embed_mode='canine_c'
echo 'Macro'
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:$memory_alloc CUDA_VISIBLE_DEVICES=$gpu_id python main.py --data $data --do_train --do_eval --embed_mode $embed_mode --batch_size $batch_size --lr $lr --epoch $epoch --seed 42 --output_file e2e_training_word_level_canine_c_avg_CONLL04_macro --eval_metric macro --mode $mode --sub_mode $sub_mode --word_pieces_aggregation avg
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:$memory_alloc CUDA_VISIBLE_DEVICES=$gpu_id python main.py --data $data --do_train --do_eval --embed_mode $embed_mode --batch_size $batch_size --lr $lr --epoch $epoch --seed 42 --output_file e2e_training_word_level_canine_c_sum_CONLL04_macro --eval_metric macro --mode $mode --sub_mode $sub_mode --word_pieces_aggregation sum
echo 'Micro'
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:$memory_alloc CUDA_VISIBLE_DEVICES=$gpu_id python main.py --data $data --do_train --do_eval --embed_mode $embed_mode --batch_size $batch_size --lr $lr --epoch $epoch --seed 42 --output_file e2e_training_word_level_canine_c_avg_CONLL04_micro --eval_metric micro --mode $mode --sub_mode $sub_mode --word_pieces_aggregation avg
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:$memory_alloc CUDA_VISIBLE_DEVICES=$gpu_id python main.py --data $data --do_train --do_eval --embed_mode $embed_mode --batch_size $batch_size --lr $lr --epoch $epoch --seed 42 --output_file e2e_training_word_level_canine_c_sum_CONLL04_micro --eval_metric micro --mode $mode --sub_mode $sub_mode --word_pieces_aggregation sum

echo '--------------'
echo '--------------'
echo 'CANINE-S'
embed_mode='canine_s'
echo 'Macro'
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:$memory_alloc CUDA_VISIBLE_DEVICES=$gpu_id python main.py --data $data --do_train --do_eval --embed_mode $embed_mode --batch_size $batch_size --lr $lr --epoch $epoch --seed 42 --output_file e2e_training_word_level_canine_s_avg_CONLL04_macro --eval_metric macro --mode $mode --sub_mode $sub_mode --word_pieces_aggregation avg
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:$memory_alloc CUDA_VISIBLE_DEVICES=$gpu_id python main.py --data $data --do_train --do_eval --embed_mode $embed_mode --batch_size $batch_size --lr $lr --epoch $epoch --seed 42 --output_file e2e_training_word_level_canine_s_sum_CONLL04_macro --eval_metric macro --mode $mode --sub_mode $sub_mode --word_pieces_aggregation sum
echo 'Micro'
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:$memory_alloc CUDA_VISIBLE_DEVICES=$gpu_id python main.py --data $data --do_train --do_eval --embed_mode $embed_mode --batch_size $batch_size --lr $lr --epoch $epoch --seed 42 --output_file e2e_training_word_level_canine_s_avg_CONLL04_micro --eval_metric micro --mode $mode --sub_mode $sub_mode --word_pieces_aggregation avg
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:$memory_alloc CUDA_VISIBLE_DEVICES=$gpu_id python main.py --data $data --do_train --do_eval --embed_mode $embed_mode --batch_size $batch_size --lr $lr --epoch $epoch --seed 42 --output_file e2e_training_word_level_canine_s_sum_CONLL04_micro --eval_metric micro --mode $mode --sub_mode $sub_mode --word_pieces_aggregation sum
echo '--------------'
echo '--------------'
echo 'CharacterBERT'
embed_mode='characterBERT'
echo 'Macro'
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:$memory_alloc CUDA_VISIBLE_DEVICES=$gpu_id python main.py --data $data --do_train --do_eval --embed_mode $embed_mode --batch_size $batch_size --lr $lr --epoch $epoch --seed 42 --output_file e2e_training_word_level_characterBERT_CONLL04_macro --eval_metric macro --mode $mode --sub_mode $sub_mode
echo 'Micro'
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:$memory_alloc CUDA_VISIBLE_DEVICES=$gpu_id python main.py --data $data --do_train --do_eval --embed_mode $embed_mode --batch_size $batch_size --lr $lr --epoch $epoch --seed 42 --output_file e2e_training_word_level_characterBERT_CONLL04_micro --eval_metric micro --mode $mode --sub_mode $sub_mode