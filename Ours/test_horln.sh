basenet=elec_horln
name=elec_horln
begin_epoch=1
end_epoch=200
all_epochs=`seq $begin_epoch $end_epoch`

for epoch in $all_epochs
do

test_model=../checkpoints/electricity_"$name"/"$epoch"_net_G.pth
while  [ ! -f $test_model ]
do
        echo 'model waiting: '$test_model
        sleep 5 
done

CUDA_VISIBLE_DEVISE=0 python ../test_elec.py \
--dataset_mode electricity \
--dataroot ../datasets/electricity/ \
--batch_size 1 \
--num_test 100000 \
--model electricity \
--netG $basenet \
--name electricity_"$name" \
--epoch $epoch
done