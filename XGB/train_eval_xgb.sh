net=xgb
name=xgb

python -u train_eval_xgb.py \
--dataset_mode electricity \
--dataroot ../datasets/electricity/ \
--batch_size 1 \
--model electricity \
--name electricity_"$name"
