#!/usr/bin/env bash

sets=0
gpus=$1
up=$2
data_dir=$5
case $3 in
  iLiDS-VID)
    main=dqn.py
    base=ilds_$4_$sets
    num_id=150
    train_set=image_valid$sets
    valid_set=image_test$sets
    ;;
  PRID-2011)
    main=dqn.py
    base=prid_$4_$sets
    num_id=100
    train_set=image_valid$sets
    valid_set=image_test$sets
    ;;
  MARS)
    main=dqn_mars.py
    base=mars_$4
    num_id=624
    train_set=image_valid
    valid_set=image_test
    ;;
  *)
    echo "No valid dataset"
    exit
    ;;
esac

bs=8
ss=8
ms=$ss
lr=1e-4
epochs=100
ts=$(date "+%Y.%m.%d-%H.%M.%S")
qg=0.9
nh=128
ns=32
mode=DQN_test-$sets-$ts-bs$bs-ss$ss-$4

python $main --gpus $gpus --data-dir $data_dir \
--num-examples 100000 --num-id $num_id \
--train-set $train_set --valid-set $valid_set \
--sample-size $ss --batch-size $bs \
--lr $lr --num-epoches $epochs --mode $3-TEST-$mode \
--model-load-epoch 1 --model-load-prefix $base --q-gamma $qg \
--penalty $up --num_hidden $nh --num_sim $ns \
--min-states $ms --optimizer sgd \
--epsilon --e2e --fusion