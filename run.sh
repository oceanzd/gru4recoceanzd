#! /bin/bash

# 十分之一数据
yoochoose_data_path=/home/ljm/dataset/yoochoose/sampled-one-tenth
yoochoose_result_path=/home/ljm/var/result/yoochoose

train_path=$yoochoose_data_path/rsc15_train_tr.txt
test_path=$yoochoose_data_path/rsc15_train_valid.txt
result_path=$yoochoose_result_path/gru4rec.csv

for rnn_size in 64 128 256; do
for batch_size in 64 128 256; do
for hidden_act in tanh; do
for dropout_p_hidden in 0.5; do
for final_act in tanh; do
for loss in top1; do
for optimizer in adam; do
for lr in 0.0005 0.001 0.005 0.01 0.05; do
for decay in 0.999;do
for n_sample in 0; do
for sample_alpha in 0.5; do
for train_random_order in False; do
for time_sort in False; do
for is_training in 1; do

#for rnn_size in 200; do
#for batch_size in 256; do
#for hidden_act in tanh; do
#for dropout_p_hidden in 0.3; do
#for final_act in tanh; do
#for loss in top1; do
#for optimizer in adam; do
#for lr in 0.005; do
#for decay in 0.999;do
#for n_sample in 0; do
#for sample_alpha in 0.5; do
#for train_random_order in False; do
#for time_sort in False; do
#for is_training in 1; do

#echo -e "rnn_size: $rnn_size \011 batch_size: $batch_size \011 dropout_p_hidden: $dropout_p_hidden \011 lr: $lr \011 optimizer: $optimizer"

python3 main.py --train $train_path --test $test_path \
     --is_training $is_training --reset_after_session True --n_epochs 50 --rnn_size $rnn_size --batch_size $batch_size \
     --hidden_act $hidden_act --dropout_p_hidden $dropout_p_hidden --final_act $final_act --loss $loss \
     --lr $lr --decay $decay --optimizer $optimizer --n_sample $n_sample --sample_alpha $sample_alpha \
     --train_random_order $train_random_order --time_sort $time_sort


done
done
done
done
done
done
done
done
done
done
done
done
done
done
