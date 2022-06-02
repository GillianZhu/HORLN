net=elec_cnn
name=elec_cnn

CUDA_VISIBLE_DEVICES=0 nohup python ../train_elec.py \
--dataset_mode electricity \
--dataroot ../datasets/electricity/ \
--batch_size 32 \
--lr 0.0001 \
--model electricity \
--netG $net \
--name electricity_"$name" \
--save_epoch_freq 1
