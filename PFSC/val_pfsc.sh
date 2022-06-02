basenet=pfsc
name=pfsc
end_epoch=201

CUDA_VISIBLE_DEVICES=0 python val_pfsc.py \
--phase val \
--dataset_mode electricity \
--dataroot ../datasets/electricity_"$name"/val/ \
--batch_size 1 \
--num_test 100000 \
--model electricity \
--netG $basenet \
--name electricity_"$name" \
--epoch $end_epoch