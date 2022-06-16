#!/bin/bash

memory_alloc=32
gpu_id=0
data='CONLL04'
epoch=200
batch_size=10
lr=0.00002
clip=1
mode='pretrained_frozen_emb'
sub_mode='word_level'
offline_emb_path='offline_embeddings/'

cd ..
cd ..
echo 'BERT cased'
echo 'Evaluation: Micro'
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:$memory_alloc CUDA_VISIBLE_DEVICES=$gpu_id python main.py --data $data --do_train --do_eval --embed_mode bert_cased --batch_size $batch_size --lr $lr --clip $clip --epoch $epoch --seed 42 --output_file pretrained_frozen_emb_bert_avg_CONLL04_micro --eval_metric micro --mode $mode --sub_mode $sub_mode --word_pieces_aggregation avg --offline_emb_path $offline_emb_path
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:$memory_alloc CUDA_VISIBLE_DEVICES=$gpu_id python main.py --data $data --do_train --do_eval --embed_mode bert_cased --batch_size $batch_size --lr $lr --clip $clip --epoch $epoch --seed 42 --output_file pretrained_frozen_emb_bert_sum_CONLL04_micro --eval_metric micro --mode $mode --sub_mode $sub_mode --word_pieces_aggregation sum --offline_emb_path $offline_emb_path
echo 'Evaluation: Macro'
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:$memory_alloc CUDA_VISIBLE_DEVICES=$gpu_id python main.py --data $data --do_train --do_eval --embed_mode bert_cased --batch_size $batch_size --lr $lr --clip $clip --epoch $epoch --seed 42 --output_file pretrained_frozen_emb_bert_avg_CONLL04_macro --eval_metric macro --mode $mode --sub_mode $sub_mode --word_pieces_aggregation avg --offline_emb_path $offline_emb_path
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:$memory_alloc CUDA_VISIBLE_DEVICES=$gpu_id python main.py --data $data --do_train --do_eval --embed_mode bert_cased --batch_size $batch_size --lr $lr --clip $clip --epoch $epoch --seed 42 --output_file pretrained_frozen_emb_bert_sum_CONLL04_macro --eval_metric macro --mode $mode --sub_mode $sub_mode --word_pieces_aggregation sum --offline_emb_path $offline_emb_path
echo '###################################################################################################################'
echo 'ALBERT'
echo 'Evaluation: Micro'
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:$memory_alloc CUDA_VISIBLE_DEVICES=$gpu_id python main.py --data $data --do_train --do_eval --embed_mode albert --batch_size $batch_size --lr $lr --clip $clip --epoch $epoch --seed 42 --output_file pretrained_frozen_emb_albert_avg_CONLL04_micro --eval_metric micro --mode $mode --sub_mode $sub_mode --word_pieces_aggregation avg --offline_emb_path $offline_emb_path
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:$memory_alloc CUDA_VISIBLE_DEVICES=$gpu_id python main.py --data $data --do_train --do_eval --embed_mode albert --batch_size $batch_size --lr $lr --clip $clip --epoch $epoch --seed 42 --output_file pretrained_frozen_emb_albert_sum_CONLL04_micro --eval_metric micro --mode $mode --sub_mode $sub_mode --word_pieces_aggregation sum --offline_emb_path $offline_emb_path
echo 'Evaluation: Macro'
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:$memory_alloc CUDA_VISIBLE_DEVICES=$gpu_id python main.py --data $data --do_train --do_eval --embed_mode albert --batch_size $batch_size --lr $lr --clip $clip --epoch $epoch --seed 42 --output_file pretrained_frozen_emb_albert_avg_CONLL04_macro --eval_metric macro --mode $mode --sub_mode $sub_mode --word_pieces_aggregation avg --offline_emb_path $offline_emb_path
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:$memory_alloc CUDA_VISIBLE_DEVICES=$gpu_id python main.py --data $data --do_train --do_eval --embed_mode albert --batch_size $batch_size --lr $lr --clip $clip --epoch $epoch --seed 42 --output_file pretrained_frozen_emb_albert_sum_CONLL04_macro --eval_metric macro --mode $mode --sub_mode $sub_mode --word_pieces_aggregation sum --offline_emb_path $offline_emb_path
echo '###################################################################################################################'
echo 'characterBERT'
echo 'Evaluation: Micro'
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:$memory_alloc CUDA_VISIBLE_DEVICES=$gpu_id python main.py --data $data --do_train --do_eval --embed_mode characterBERT --batch_size $batch_size --lr $lr --clip $clip --epoch $epoch --seed 42 --output_file pretrained_frozen_emb_characterBERT_CONLL04_micro --eval_metric micro --mode $mode --sub_mode $sub_mode --offline_emb_path $offline_emb_path
echo 'Evaluation: Macro'
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:$memory_alloc CUDA_VISIBLE_DEVICES=$gpu_id python main.py --data $data --do_train --do_eval --embed_mode characterBERT --batch_size $batch_size --lr $lr --clip $clip --epoch $epoch --seed 42 --output_file pretrained_frozen_emb_characterBERT_CONLL04_macro --eval_metric macro --mode $mode --sub_mode $sub_mode --offline_emb_path $offline_emb_path
echo '###################################################################################################################'
echo 'CANINE-c'
echo 'Evaluation: Micro'
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:$memory_alloc CUDA_VISIBLE_DEVICES=$gpu_id python main.py --data $data --do_train --do_eval --embed_mode canine_c --batch_size $batch_size --lr $lr --clip $clip --epoch $epoch --seed 42 --output_file pretrained_frozen_emb_canine_c_avg_CONLL04_micro --eval_metric micro --mode $mode --sub_mode $sub_mode --word_pieces_aggregation avg --offline_emb_path $offline_emb_path
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:$memory_alloc CUDA_VISIBLE_DEVICES=$gpu_id python main.py --data $data --do_train --do_eval --embed_mode canine_c --batch_size $batch_size --lr $lr --clip $clip --epoch $epoch --seed 42 --output_file pretrained_frozen_emb_canine_c_sum_CONLL04_micro --eval_metric micro --mode $mode --sub_mode $sub_mode --word_pieces_aggregation sum --offline_emb_path $offline_emb_path
echo 'Evaluation: Macro'
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:$memory_alloc CUDA_VISIBLE_DEVICES=$gpu_id python main.py --data $data --do_train --do_eval --embed_mode canine_c --batch_size $batch_size --lr $lr --clip $clip --epoch $epoch --seed 42 --output_file pretrained_frozen_emb_canine_c_avg_CONLL04_macro --eval_metric macro --mode $mode --sub_mode $sub_mode --word_pieces_aggregation avg --offline_emb_path $offline_emb_path
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:$memory_alloc CUDA_VISIBLE_DEVICES=$gpu_id python main.py --data $data --do_train --do_eval --embed_mode canine_c --batch_size $batch_size --lr $lr --clip $clip --epoch $epoch --seed 42 --output_file pretrained_frozen_emb_canine_c_sum_CONLL04_macro --eval_metric macro --mode $mode --sub_mode $sub_mode --word_pieces_aggregation sum --offline_emb_path $offline_emb_path
echo '###################################################################################################################'
echo 'CANINE-s'
echo 'Evaluation: Micro'
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:$memory_alloc CUDA_VISIBLE_DEVICES=$gpu_id python main.py --data $data --do_train --do_eval --embed_mode canine_s --batch_size $batch_size --lr $lr --clip $clip --epoch $epoch --seed 42 --output_file pretrained_frozen_emb_canine_s_avg_CONLL04_micro --eval_metric micro --mode $mode --sub_mode $sub_mode --word_pieces_aggregation avg --offline_emb_path $offline_emb_path
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:$memory_alloc CUDA_VISIBLE_DEVICES=$gpu_id python main.py --data $data --do_train --do_eval --embed_mode canine_s --batch_size $batch_size --lr $lr --clip $clip --epoch $epoch --seed 42 --output_file pretrained_frozen_emb_canine_s_sum_CONLL04_micro --eval_metric micro --mode $mode --sub_mode $sub_mode --word_pieces_aggregation sum --offline_emb_path $offline_emb_path
echo 'Evaluation: Macro'
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:$memory_alloc CUDA_VISIBLE_DEVICES=$gpu_id python main.py --data $data --do_train --do_eval --embed_mode canine_s --batch_size $batch_size --lr $lr --clip $clip --epoch $epoch --seed 42 --output_file pretrained_frozen_emb_canine_s_avg_CONLL04_macro --eval_metric macro --mode $mode --sub_mode $sub_mode --word_pieces_aggregation avg --offline_emb_path $offline_emb_path
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:$memory_alloc CUDA_VISIBLE_DEVICES=$gpu_id python main.py --data $data --do_train --do_eval --embed_mode canine_s --batch_size $batch_size --lr $lr --clip $clip --epoch $epoch --seed 42 --output_file pretrained_frozen_emb_canine_s_sum_CONLL04_macro --eval_metric macro --mode $mode --sub_mode $sub_mode --word_pieces_aggregation sum --offline_emb_path $offline_emb_path
echo '###################################################################################################################'