#!/bin/bash

memory_alloc=32
gpu_id=0
data='CONLL04'
epoch=200
batch_size=10
lr=0.00002
clip=1
mode='e2e_training'

cd ..
cd ..
echo 'BERT cased'
embed_mode='bert_cased'
echo 'Macro'
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:$memory_alloc CUDA_VISIBLE_DEVICES=$gpu_id python main.py --data $data --do_train --do_eval --embed_mode $embed_mode --batch_size $batch_size --lr $lr --epoch $epoch --seed 42 --output_file e2e_training_bert_head_head_CONLL04_macro --eval_metric macro --mode $mode
echo 'Micro'
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:$memory_alloc CUDA_VISIBLE_DEVICES=$gpu_id python main.py --data $data --do_train --do_eval --embed_mode $embed_mode --batch_size $batch_size --lr $lr --epoch $epoch --seed 42 --output_file e2e_training_bert_head_head_CONLL04_micro --eval_metric micro --mode $mode
echo '--------------'
echo '--------------'
echo 'ALBERT'
embed_mode='albert'
echo 'Macro'
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:$memory_alloc CUDA_VISIBLE_DEVICES=$gpu_id python main.py --data $data --do_train --do_eval --embed_mode $embed_mode --batch_size $batch_size --lr $lr --epoch $epoch --seed 42 --output_file e2e_training_albert_head_head_CONLL04_macro --eval_metric macro --mode $mode
echo 'Micro'
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:$memory_alloc CUDA_VISIBLE_DEVICES=$gpu_id python main.py --data $data --do_train --do_eval --embed_mode $embed_mode --batch_size $batch_size --lr $lr --epoch $epoch --seed 42 --output_file e2e_training_albert_head_head_CONLL04_micro --eval_metric micro --mode $mode