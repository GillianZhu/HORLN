name=elec_horln
epoch=1

python ../eval_elec.py \
--dataset_mode electricity \
--dataroot ../datasets/electricity/ \
--batch_size 1 \
--num_test 100000 \
--model electricity \
--name electricity_"$name" \
--epoch $epoch \
--best_threshold 0.0