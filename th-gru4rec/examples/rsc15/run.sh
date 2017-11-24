#! /bin/bash

for rnn_size in 64 128 256; do
for batch_size in 64 128 256; do
for hidden_act in tanh; do
for dropout_p_hidden in 0.5; do
for final_act in tanh; do
for loss in top1; do
for optimizer in adam adagrad; do

#for rnn_size in 256; do
#for batch_size in 256; do
#for hidden_act in tanh; do
#for dropout_p_hidden in 0.5; do
#for final_act in tanh; do
#for loss in top1; do
#for optimizer in adam; do

if [ $optimizer == adam ]; then
  for learning_rate in 0.0005 0.001 0.005; do
    python3 main.py  --n_epochs 50 --rnn_size $rnn_size --batch_size $batch_size --hidden_act $hidden_act --dropout_p_hidden $dropout_p_hidden --final_act $final_act --loss $loss  --learning_rate $learning_rate --optimizer $optimizer
  done
elif [ $optimizer == adagrad ]; then
  for learning_rate in 0.005 0.01 0.05; do
    python3 main.py  --n_epochs 50 --rnn_size $rnn_size --batch_size $batch_size --hidden_act $hidden_act --dropout_p_hidden $dropout_p_hidden --final_act $final_act --loss $loss  --learning_rate $learning_rate --optimizer $optimizer
  done
fi
done
done
done
done
done
done
done
