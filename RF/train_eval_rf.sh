net=rf
name=rf

python -u train_eval_rf.py \
--dataset_mode electricity \
--dataroot ../datasets/electricity/ \
--batch_size 1 \
--model electricity \
--name electricity_"$name"
