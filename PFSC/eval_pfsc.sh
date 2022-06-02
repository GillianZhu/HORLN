name=pfsc
epoch=2

python ../eval_elec.py \
--dataset_mode electricity \
--dataroot ../datasets/electricity/ \
--batch_size 1 \
--num_test 100000 \
--name electricity_"$name" \
--model electricity \
--epoch $epoch \
--best_threshold 0.0