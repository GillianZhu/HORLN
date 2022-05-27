name=elec_horln
epoch=2

python eval.py \
--dataset_mode electricity \
--dataroot ./datasets/electricity/ \
--batch_size 1 \
--num_test 100000 \
--model electricity \
--name electricity_"$name" \
--epoch $epoch \
--best_threshold 0.60579664