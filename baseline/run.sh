#!/usr/bin/env bash

bs=8
epochs=1
sets=0
gpus=$1
data_dir=$4
case $2 in
  iLiDS-VID)
    base=ilds_$3_$sets
    num_id=150
    train_set=image_valid$sets
    valid_set=image_test$sets
    ;;
  PRID-2011)
    base=prid_$3_$sets
    num_id=100
    train_set=image_valid$sets
    valid_set=image_test$sets
    ;;
  MARS)
    base=mars_$3
    num_id=624
    train_set=image_valid
    valid_set=image_test
    ;;
  *)
    echo "No valid dataset"
    exit
    ;;
esac

case $3 in
  alexnet)
    python baseline.py --gpus $gpus --data-dir $data_dir \
        --num-id $num_id --batch-size $bs \
        --train-set $train_set --valid-set $valid_set \
        --lr 1e-4 --num-epoches $epochs --mode $mode \
        --network alexnet --model-load-prefix alexnet --model-load-epoch 1
    ;;
  inception-bn)
     python baseline.py --gpus $gpus --data-dir $data_dir \
        --num-id $num_id --batch-size $bs \
        --train-set $train_set --valid-set $valid_set \
        --lr 1e-2 --num-epoches $epochs --mode $mode --lsoftmax
    ;;
  *)
    echo "No valid basenet"
    exit
    ;;
esac
