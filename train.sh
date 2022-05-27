net=elec_horln
name=elec_horln

python train.py \
--dataset_mode electricity \
--dataroot ./datasets/electricity/ \
--batch_size 32 \
--lr 0.0001 \
--model electricity \
--netG $net \
--name electricity_"$name" \
--save_epoch_freq 1
