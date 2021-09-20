#!/usr/bin/env bash

export TRAIN_FILE=../../../data/uniform-prior-symbolic-format/train.txt
export TEST_FILE=../../../data/uniform-prior-symbolic-format/test.txt

python train_t5_end.py \
    --output_dir=experiment_result \ #train_arg
    --model_type=t5 \ #model_arg
    --tokenizer_name=t5-large \ #model_arg
    --model_name_or_path=symtime-pretrained-model/start \ #model_arg
    --duration_model_path=symtime-pretrained-model/duration \ #model_arg
    --do_train \ #train_arg
    --do_eval \ #train_arg
    --num_train_epochs=50 \ #train_arg
    --train_data_file=$TRAIN_FILE \ #data_arg
    --eval_data_file=$TEST_FILE \ #data_arg
    --line_by_line \ #data_arg
    --per_gpu_train_batch_size=8 \ #train_arg
    --per_device_train_batch_size=8 \ #train_arg
    --gradient_accumulation_steps=4 \ #train_arg
    --per_device_eval_batch_size=8 \ #train_arg
    --per_gpu_eval_batch_size=8 \ #train_arg
    --save_steps=5000 \ #train_arg
    --logging_steps=10 \ #train_arg
    --overwrite_output_dir \ #train_arg
    --seed=10 \ #train_arg

# train_t5_end.py --output_dir=experiment_result --model_type=t5 --tokenizer_name=t5-large --model_name_or_path=symtime-pretrained-model/start --duration_model_path=symtime-pretrained-model/duration --do_train --do_eval --num_train_epochs=50 --train_data_file=../../../data/uniform-prior-symbolic-format/train.txt --eval_data_file=../../../data/uniform-prior-symbolic-format/test.txt
# --line_by_line --per_gpu_train_batch_size=8 --per_device_train_batch_size=8 --gradient_accumulation_steps=4 --per_device_eval_batch_size=8 --per_gpu_eval_batch_size=8 --save_steps=5000 --logging_steps=10 --overwrite_output_dir --seed=10 