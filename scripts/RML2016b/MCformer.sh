export CUDA_VISIBLE_DEVICES=0

# The model name and training dataset
model=MCformer
dataset=RML2016b

# The split ratio for the training dataset
split_ratio=0.6

for snr in {-20..18..2}
do
  for batch_size in 32 64 16 
  do
    for learning_rate in 0.0001 0.00005 0.000025 0.00001
    do
    #   clear
      echo "model: $model, dataset: $dataset, SNR: $snr, batch_size: $batch_size, learning_rate: $learning_rate"
      python main.py \
        --model $model \
        --dataset $dataset \
        --snr $snr \
        --file_path /root/autodl-tmp/dataset/RML2016.10b.dat \
        --batch_size $batch_size \
        --num_epochs 64 \
        --learning_rate $learning_rate \
        --optimizer adam \
        --criterion cross_entropy \
        --split_ratio $split_ratio \
        --warmup_epochs 0 \
        --d_model 128 \
        --d_ff 256 \
        --n_layers 3 \
        --patience 10
    done
  done
done