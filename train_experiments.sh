
case_slice_0=(82 119 47 65 66 43)
case_slice_1=(35 30 33 34)


#python train.py --name ep12_new_bz16_semi_cps --epoch_subsample 100 --select_epoch 12
#python train.py --name ep12_new_bz16_semi_entropy_50_cps --epoch_subsample 50 --select_epoch 12

#python train.py --dataroot ../../datasets/brats --checkpoints_dir brats_checkpoints/New_Snapshots_T1 \
#  --dataset_mode unalignbrats --model unet_cps_brats --target t1 --num_slice 155 --name "w|o_adaptaion" \
#  --log_dir brats_logs/val_logs/t1 --data_input train
#
#python test.py --dataroot ../../datasets/brats --checkpoints_dir brats_checkpoints/New_Snapshots_T1 \
#  --dataset_mode unalignbrats_test --model test_brats --target t1 --num_slice 155 --name "w|o_adaptaion" \
#  --log_dir brats_logs/test_logs/t1 --data_input test --epoch 'best'
#
#python train.py --dataroot ../../datasets/brats --checkpoints_dir brats_checkpoints/New_Snapshots_FLAIR \
#  --dataset_mode unalignbrats --model unet_cps --target flair --num_slice 155 --name "adaptaion" \
#  --log_dir brats_log/val_logs/flair --data_input train --gpu_ids 2 --num_epoch 100 \
#  --up_epoch 50 --down_epoch 50 --epoch_subsample 101
#
#python test.py --dataroot ../../datasets/brats --checkpoints_dir brats_checkpoints/New_Snapshots_T1CE \
#  --dataset_mode unalignbrats_test --model test_brats --target t1ce --num_slice 155 --name "w|o_adaptaion" \
#  --log_dir brats_logs/test_logs/t1ce --data_input test --epoch 'best'
#
#python train.py --dataroot ../../datasets/brats --checkpoints_dir brats_checkpoints/New_Snapshots_FLAIR \
#  --dataset_mode unalignbrats --model unet_cps_brats --target flair --num_slice 155 --name "w|o_adaptaion" \
#  --log_dir brats_logs/val_logs/flair --data_input train
#
#python test.py --dataroot ../../datasets/brats --checkpoints_dir brats_checkpoints/New_Snapshots_FLAIR \
#  --dataset_mode unalignbrats_test --model test_brats --target flair --num_slice 155 --name "w|o_adaptaion" \
#  --log_dir brats_logs/test_logs/flair --data_input test --epoch 'best'


####abdominal segmentation
#  python train.py --dataroot ../../datasets/abdominal --checkpoints_dir abdo_checkpoints/New_Snapshots_t22ct --name "ep50_new_bz16_semi_cps"\
#   --model unet_cps_abdo --direction BtoA --phase train --data_input train --dataset unalignabdo --select_epoch 50 --case_slice ${case_slice_0[*]} \
#   --num_epoch 200 --up_epoch 100 --down_epoch 100 --epoch_subsample 201 --log_dir abdo_logs/val_logs/t22ct --gpu_ids 2
##
#  python test.py --dataroot ../../datasets/abdominal --checkpoints_dir abdo_checkpoints/New_Snapshots_t12ct --name "w|o_adaptation_semi_cps"\
#    --model test_abdo --direction BtoA --case_slice ${case_slice_0[*]} --phase test --dataset unalignabdo_test \
#    --log_dir abdo_log/test_logs/t12ct --epoch 'best' --log_dir abdo_logs/test_logs/t12ct

##WHS segmentation

#python train.py --dataroot ../../datasets/2017-MM-WHS/data --checkpoints_dir whs_checkpoints/New_Snapshots_mr2ct --name "w|o_adaptation_semi_cps"\
#   --model unet_cps --direction BtoA --phase train --data_input train --dataset unalignwhs --select_epoch 50 --num_slice 600 --no_adaptation 1 \
#   --num_epoch 100 --up_epoch 50 --down_epoch 50 --epoch_subsample 101 --log_dir whs_logs/val_logs/mr2ct

#  python test.py --dataroot ../../datasets/2017-MM-WHS/data --checkpoints_dir whs_checkpoints/New_Snapshots_mr2ct --name "w|o_adaptation_semi_cps"\
#    --model test --direction BtoA --num_slice 256 --phase test --dataset unalignwhs_test \
#    --log_dir whs_log/test_logs/mr2ct --epoch 'best'


#python train.py --dataroot ../../datasets/2017-MM-WHS/data --checkpoints_dir whs_checkpoints/New_Snapshots_ct2mr --name "w|o_adaptation_semi_cps"\
#   --model unet_cps --direction AtoB --phase train --data_input train --dataset unalignwhs --select_epoch 50 --num_slice 600 --no_adaptation 1 \
#   --num_epoch 100 --up_epoch 50 --down_epoch 50 --epoch_subsample 101 --log_dir whs_logs/val_logs/ct2mr

#python train.py --dataroot ../../datasets/2017-MM-WHS/data --checkpoints_dir whs_checkpoints/New_Snapshots_mr2ct --name "w|o_adaptation"\
#   --model unet_cps --direction BtoA --phase train --data_input train --dataset unalignwhs --select_epoch 12 --num_slice 600 \
#   --num_epoch 100 --up_epoch 50 --down_epoch 50 --epoch_subsample 101 --log_dir whs_logs/val_logs/mr2ct --no_adaptation 1 \
#   --gpu_ids 1

#python train.py --dataroot ../../datasets/2017-MM-WHS/data --checkpoints_dir whs_checkpoints/New_Snapshots_mr2ct --name "ep12_new_bz16"\
#   --model unet_cps --direction BtoA --phase train --data_input train --dataset unalignwhs --select_epoch 12 --num_slice 600 \
#   --num_epoch 100 --up_epoch 50 --down_epoch 50 --epoch_subsample 101 --log_dir whs_logs/val_logs/mr2ct --no_adaptation 1\
#   --gpu_ids 3


#python train.py --dataroot ../../datasets/2017-MM-WHS/data --checkpoints_dir whs_checkpoints/New_Snapshots_mr2ct --name "ep12_new_bz16_semi_cps"\
#   --model unet_cps --direction BtoA --phase train --data_input train --dataset unalignwhs --select_epoch 12 --num_slice 600 \
#   --num_epoch 100 --up_epoch 50 --down_epoch 50 --epoch_subsample 101 --log_dir whs_logs/val_logs/mr2ct \
#   --gpu_ids 0
#

#python train.py --dataroot ../../datasets/2017-MM-WHS/data --checkpoints_dir whs_checkpoints/New_Snapshots_mr2ct --name "ep12_new_bz16_semi_entropy_50_cps_new"\
#   --model unet_cps --direction BtoA --phase train --data_input train --dataset unalignwhs --select_epoch 12 --num_slice 600 \
#   --num_epoch 100 --up_epoch 50 --down_epoch 50 --epoch_subsample 50 --log_dir whs_logs/val_logs/mr2ct \
#   --gpu_ids 2

#python train.py --dataroot ../../datasets/2017-MM-WHS/data --checkpoints_dir whs_checkpoints/New_Snapshots_ct2mr --name "ep12_new_bz16_semi_entropy_60_cps_new"\
#   --model unet_cps --direction AtoB --phase train --data_input train --dataset unalignwhs --select_epoch 12 --num_slice 600 \
#   --num_epoch 100 --up_epoch 50 --down_epoch 50 --epoch_subsample 60 --log_dir whs_logs/val_logs/ct2mr \
#   --gpu_ids 0


#python train.py --dataroot ../../datasets/2017-MM-WHS/data --checkpoints_dir whs_checkpoints/New_Snapshots_mr2ct --name "ep12_new_bz16_semi_entropy_40_cps"\
#   --model unet_cps --direction BtoA --phase train --data_input train --dataset unalignwhs --select_epoch 12 --num_slice 600 \
#   --num_epoch 100 --up_epoch 50 --down_epoch 50 --epoch_subsample 40 --log_dir whs_logs/val_logs/mr2ct \
#   --gpu_ids 2


#python train.py --dataroot ../../datasets/2017-MM-WHS/data --checkpoints_dir whs_checkpoints/New_Snapshots_ct2mr --name "ep12_new_bz16_semi_entropy_60_cps"\
#   --model unet_cps --direction AtoB --phase train --data_input train --dataset unalignwhs --select_epoch 12 --num_slice 600 \
#   --num_epoch 100 --up_epoch 50 --down_epoch 50 --epoch_subsample 60 --log_dir whs_logs/val_logs/ct2mr \
#   --gpu_ids 0


#python train.py --dataroot ../../datasets/2017-MM-WHS/data --checkpoints_dir whs_checkpoints/New_Snapshots_ct2mr --name "ep12_new_bz16_semi_cps"\
#   --model unet_cps --direction AtoB --phase train --data_input train --dataset unalignwhs --select_epoch 12 --num_slice 600 \
#   --num_epoch 100 --up_epoch 50 --down_epoch 50 --epoch_subsample 101 --log_dir whs_logs/val_logs/ct2mr \
#   --gpu_ids 0

#python train.py --dataroot ../../datasets/2017-MM-WHS/data --checkpoints_dir whs_checkpoints/New_Snapshots_ct2mr --name "ep12_new_bz16_semi_entropy_50_cps"\
#   --model unet_cps --direction AtoB --phase train --data_input train --dataset unalignwhs --select_epoch 12 --num_slice 600 \
#   --num_epoch 100 --up_epoch 50 --down_epoch 50 --epoch_subsample 50 --log_dir whs_logs/val_logs/ct2mr \
#   --gpu_ids 1


#python train.py --dataroot ../../datasets/2017-MM-WHS/data --checkpoints_dir whs_checkpoints/New_Snapshots_mr2ct --name "ep12_new_bz16_semi_entropy_50_cps"\
#   --model unet_cps --direction BtoA --phase train --data_input train --dataset unalignwhs --select_epoch 12 --num_slice 600 \
#   --num_epoch 100 --up_epoch 50 --down_epoch 50 --epoch_subsample 50 --log_dir whs_logs/val_logs/mr2ct \
#   --gpu_ids 2



#python train.py --dataroot ../../datasets/2017-MM-WHS/data --checkpoints_dir whs_checkpoints/New_Snapshots_ct2mr --name "ep12_new_bz16"\
#   --model unet_cps --direction AtoB --phase train --data_input train --dataset unalignwhs --select_epoch 12 --num_slice 600 \
#   --num_epoch 100 --up_epoch 50 --down_epoch 50 --epoch_subsample 101 --log_dir whs_logs/val_logs/ct2mr \
#   --gpu_ids 1


#python train.py --dataroot ../../datasets/2017-MM-WHS/data --checkpoints_dir whs_checkpoints/New_Snapshots_ct2mr --name "ep12_new_bz16"\
#   --model unet_cps --direction AtoB --phase train --data_input train --dataset unalignwhs --select_epoch 12 --num_slice 600 \
#   --num_epoch 100 --up_epoch 50 --down_epoch 50 --epoch_subsample 0 --log_dir whs_logs/val_logs/ct2mr \
#   --gpu_ids 1
#
#
#python train.py --dataroot ../../datasets/2017-MM-WHS/data --checkpoints_dir whs_checkpoints/New_Snapshots_ct2mr --name "ep12_new_bz16_semi_cps"\
#   --model unet_cps --direction AtoB --phase train --data_input train --dataset unalignwhs --select_epoch 12 --num_slice 600 \
#   --num_epoch 100 --up_epoch 50 --down_epoch 50 --epoch_subsample 101 --log_dir whs_logs/val_logs/ct2mr \
#   --gpu_ids 1
#
#
#python train.py --dataroot ../../datasets/2017-MM-WHS/data --checkpoints_dir whs_checkpoints/New_Snapshots_ct2mr --name "ep12_new_bz16_semi_entropy_50_cps"\
#   --model unet_cps --direction AtoB --phase train --data_input train --dataset unalignwhs --select_epoch 12 --num_slice 600 \
#   --num_epoch 100 --up_epoch 50 --down_epoch 50 --epoch_subsample 50 --log_dir whs_logs/val_logs/ct2mr \
#   --gpu_ids 2

#  python test.py --dataroot ../../datasets/2017-MM-WHS/data --checkpoints_dir whs_checkpoints/New_Snapshots_ct2mr --name "w|o_adaptation_semi_cps"\
#    --model test --direction AtoB --num_slice 130 --phase test --dataset unalignwhs_test \
#    --log_dir whs_log/test_logs/mr2ct --epoch 'best'

#  python train.py --dataroot ../../datasets/abdominal --checkpoints_dir abdo_checkpoints/New_Snapshots_t22ct --name "ep100_new_bz16_semi_cps"\
#   --model unet_cps_abdo --direction BtoA --phase train --data_input train --dataset unalignabdo --select_epoch 100 --case_slice ${case_slice_0[*]} \
#   --num_epoch 200 --up_epoch 100 --down_epoch 100 --epoch_subsample 201 --log_dir abdo_logs/val_logs/t22ct --gpu_ids 0

  python train.py --dataroot ../../datasets/abdominal --checkpoints_dir abdo_checkpoints/New_Snapshots_t22ct --name "ep50_new_bz16_semi_entropy_120_cps"\
   --model unet_cps --direction BtoA --phase train --data_input train --dataset unalignabdo --select_epoch 50 --case_slice ${case_slice_0[*]} \
   --num_epoch 200 --up_epoch 100 --down_epoch 100 --epoch_subsample 120 --log_dir abdo_logs/val_logs/t22ct --gpu_ids 0
#

#  python train.py --dataroot ../../datasets/abdominal --checkpoints_dir abdo_checkpoints/New_Snapshots_ct2t2 --name "ep50_new_bz16_semi_cps"\
#   --model unet_cps_abdo --direction AtoB --phase train --data_input train --dataset unalignabdo --select_epoch 50 --case_slice ${case_slice_1[*]} \
#   --num_epoch 200 --up_epoch 100 --down_epoch 100 --epoch_subsample 201 --log_dir abdo_logs/val_logs/ct2t2 --gpu_ids 0

#  python test.py --dataroot ../../datasets/abdominal --checkpoints_dir abdo_checkpoints/New_Snapshots_t22ct --name "ep50_new_bz16_semi_cps"\
#    --model test_abdo --direction BtoA --case_slice ${case_slice_0[*]} --phase test --dataset unalignabdo_test \
#    --log_dir abdo_log/test_logs/t22ct --epoch 'best' --log_dir abdo_logs/test_logs/t22ct

#  python train.py --dataroot ../../datasets/abdominal --checkpoints_dir abdo_checkpoints/New_Snapshots_ct2t2 --name "ep50_new_bz16_semi_entropy_120_cps"\
#   --model unet_cps_abdo --direction AtoB --phase train --data_input train --dataset unalignabdo --select_epoch 50 --case_slice ${case_slice_1[*]} \
#   --num_epoch 200 --up_epoch 100 --down_epoch 100 --epoch_subsample 120 --log_dir abdo_logs/val_logs/ct2t2 --gpu_ids 0


#  python train.py --dataroot ../../datasets/abdominal --checkpoints_dir abdo_checkpoints/New_Snapshots_t22ct --name "ep100_new_bz16_semi_entropy_120_cps"\
#   --model unet_cps_abdo --direction BtoA --phase train --data_input train --dataset unalignabdo --select_epoch 100 --case_slice ${case_slice_0[*]} \
#   --num_epoch 200 --up_epoch 100 --down_epoch 100 --epoch_subsample 120 --log_dir abdo_logs/val_logs/t22ct --gpu_ids 1



