#!/bin/bash

case_slice_0=(82 119 47 65 66 43)
case_slice_1=(35 30 33 34)

#python test.py --dataroot ../../datasets/brats --checkpoints_dir brats_checkpoints/New_Snapshots_T1 --name "w|o_adaptation"\
#  --model test_brats --direction t22t1 --num_slice 155 --phase test --dataset unalignbrats_test --target t1 \
#  --log_dir brats_log/test_logs/t1 --epoch 'best'
#
#python test.py --dataroot ../../datasets/brats --checkpoints_dir brats_checkpoints/New_Snapshots_T1CE --name "w|o_adaptation"\
#  --model test_brats --direction t22t1ce --num_slice 155 --phase test --dataset unalignbrats_test --target t1ce \
#  --log_dir brats_log/test_logs/t1ce --epoch 'best'
#
#python test.py --dataroot ../../datasets/brats --checkpoints_dir brats_checkpoints/New_Snapshots_FLAIR --name "adaptation"\
#  --model test --direction t22flair --num_slice 155 --phase test --dataset unalignbrats_test --target flair \
#  --log_dir brats_log/test_logs/flair --epoch '95'
#
#python test.py --dataroot ../../datasets/brats --checkpoints_dir brats_checkpoints/New_Snapshots_FLAIR --name "adaptation"\
#  --model test --direction t22flair --num_slice 155 --phase test --dataset unalignbrats_test --target flair \
#  --log_dir brats_log/test_logs/flair --epoch '90'
#
#python test.py --dataroot ../../datasets/brats --checkpoints_dir brats_checkpoints/New_Snapshots_FLAIR --name "adaptation"\
#  --model test --direction t22flair --num_slice 155 --phase test --dataset unalignbrats_test --target flair \
#  --log_dir brats_log/test_logs/flair --epoch '85'

#  python test.py --dataroot ../../datasets/2017-MM-WHS/data --checkpoints_dir whs_checkpoints/New_Snapshots_mr2ct --name "w|o_adaptation"\
#    --model test --direction BtoA --num_slice 256 --phase test --dataset unalignwhs_test \
#    --log_dir whs_logs/test_logs/mr2ct --epoch 'best'


#  python test.py --dataroot ../../datasets/2017-MM-WHS/data --checkpoints_dir whs_checkpoints/New_Snapshots_mr2ct --name "ep12_new_bz16_semi_cps"\
#    --model test --direction BtoA --num_slice 256 --phase test --dataset unalignwhs_test \
#    --log_dir whs_logs/test_logs/mr2ct --epoch 'best' --gpu_ids 1
##
#  python test.py --dataroot ../../datasets/2017-MM-WHS/data --checkpoints_dir whs_checkpoints/New_Snapshots_mr2ct --name "ep12_new_bz16_semi_entropy_50_cps"\
#    --model test --direction BtoA --num_slice 256 --phase test --dataset unalignwhs_test \
#    --log_dir whs_logs/test_logs/mr2ct --epoch 'best' --gpu_ids 1
#
#
#  python test.py --dataroot ../../datasets/2017-MM-WHS/data --checkpoints_dir whs_checkpoints/New_Snapshots_ct2mr --name "ep12_new_bz16_semi_cps"\
#    --model test --direction AtoB --num_slice 130 --phase test --dataset unalignwhs_test \
#    --log_dir whs_logs/test_logs/ct2mr --epoch 'best' --gpu_ids 1

#  python test.py --dataroot ../../datasets/2017-MM-WHS/data --checkpoints_dir whs_checkpoints/New_Snapshots_ct2mr --name "ep12_new_bz16_semi_entropy_50_cps"\
#    --model test --direction AtoB --num_slice 130 --phase test --dataset unalignwhs_test \
#    --log_dir whs_logs/test_logs/ct2mr --epoch 'best' --gpu_ids 1
#
#
#  python test.py --dataroot ../../datasets/2017-MM-WHS/data --checkpoints_dir whs_checkpoints/New_Snapshots_mr2ct --name "ep12_new_bz16_semi_entropy_40_cps"\
#    --model test --direction BtoA --num_slice 256 --phase test --dataset unalignwhs_test \
#    --log_dir whs_logs/test_logs/mr2ct --epoch '80' --gpu_ids 1
#
#
#  python test.py --dataroot ../../datasets/2017-MM-WHS/data --checkpoints_dir whs_checkpoints/New_Snapshots_ct2mr --name "ep12_new_bz16_semi_entropy_60_cps"\
#    --model test --direction AtoB --num_slice 130 --phase test --dataset unalignwhs_test \
#    --log_dir whs_logs/test_logs/ct2mr --epoch '80' --gpu_ids 1
#
#python test.py --dataroot ../../datasets/2017-MM-WHS/data --checkpoints_dir whs_checkpoints/New_Snapshots_mr2ct --name "ep12_new_bz16_semi_entropy_40_cps"\
#    --model test --direction BtoA --num_slice 256 --phase test --dataset unalignwhs_test \
#    --log_dir whs_logs/test_logs/mr2ct --epoch '85' --gpu_ids 1
##
##
#  python test.py --dataroot ../../datasets/2017-MM-WHS/data --checkpoints_dir whs_checkpoints/New_Snapshots_ct2mr --name "ep12_new_bz16_semi_entropy_60_cps"\
#    --model test --direction AtoB --num_slice 130 --phase test --dataset unalignwhs_test \
#    --log_dir whs_logs/test_logs/ct2mr --epoch '85' --gpu_ids 1
#
#python test.py --dataroot ../../datasets/2017-MM-WHS/data --checkpoints_dir whs_checkpoints/New_Snapshots_mr2ct --name "ep12_new_bz16_semi_entropy_40_cps"\
#    --model test --direction BtoA --num_slice 256 --phase test --dataset unalignwhs_test \
#    --log_dir whs_logs/test_logs/mr2ct --epoch '90' --gpu_ids 1
##
##
#  python test.py --dataroot ../../datasets/2017-MM-WHS/data --checkpoints_dir whs_checkpoints/New_Snapshots_ct2mr --name "ep12_new_bz16_semi_entropy_60_cps"\
#    --model test --direction AtoB --num_slice 130 --phase test --dataset unalignwhs_test \
#    --log_dir whs_logs/test_logs/ct2mr --epoch '90' --gpu_ids 1
#
#
#python test.py --dataroot ../../datasets/2017-MM-WHS/data --checkpoints_dir whs_checkpoints/New_Snapshots_mr2ct --name "ep12_new_bz16_semi_entropy_40_cps"\
#    --model test --direction BtoA --num_slice 256 --phase test --dataset unalignwhs_test \
#    --log_dir whs_logs/test_logs/mr2ct --epoch '95' --gpu_ids 1
##
##
#  python test.py --dataroot ../../datasets/2017-MM-WHS/data --checkpoints_dir whs_checkpoints/New_Snapshots_ct2mr --name "ep12_new_bz16_semi_entropy_60_cps"\
#    --model test --direction AtoB --num_slice 130 --phase test --dataset unalignwhs_test \
#    --log_dir whs_logs/test_logs/ct2mr --epoch '95' --gpu_ids 1


#  python test.py --dataroot ../../datasets/2017-MM-WHS/data --checkpoints_dir whs_checkpoints/New_Snapshots_mr2ct --name "ep12_new_bz16_semi_entropy_50_cps"\
#    --model test --direction BtoA --num_slice 256 --phase test --dataset unalignwhs_test \
#    --log_dir whs_logs/test_logs/mr2ct --epoch '80' --gpu_ids 1
#
#
#  python test.py --dataroot ../../datasets/2017-MM-WHS/data --checkpoints_dir whs_checkpoints/New_Snapshots_ct2mr --name "ep12_new_bz16_semi_entropy_50_cps"\
#    --model test --direction AtoB --num_slice 130 --phase test --dataset unalignwhs_test \
#    --log_dir whs_logs/test_logs/ct2mr --epoch '80' --gpu_ids 1
#
#python test.py --dataroot ../../datasets/2017-MM-WHS/data --checkpoints_dir whs_checkpoints/New_Snapshots_mr2ct --name "ep12_new_bz16_semi_entropy_50_cps"\
#    --model test --direction BtoA --num_slice 256 --phase test --dataset unalignwhs_test \
#    --log_dir whs_logs/test_logs/mr2ct --epoch '85' --gpu_ids 1
##
##
#  python test.py --dataroot ../../datasets/2017-MM-WHS/data --checkpoints_dir whs_checkpoints/New_Snapshots_ct2mr --name "ep12_new_bz16_semi_entropy_50_cps"\
#    --model test --direction AtoB --num_slice 130 --phase test --dataset unalignwhs_test \
#    --log_dir whs_logs/test_logs/ct2mr --epoch '85' --gpu_ids 1
#
#python test.py --dataroot ../../datasets/2017-MM-WHS/data --checkpoints_dir whs_checkpoints/New_Snapshots_mr2ct --name "ep12_new_bz16_semi_entropy_50_cps"\
#    --model test --direction BtoA --num_slice 256 --phase test --dataset unalignwhs_test \
#    --log_dir whs_logs/test_logs/mr2ct --epoch '90' --gpu_ids 1
##
##
#  python test.py --dataroot ../../datasets/2017-MM-WHS/data --checkpoints_dir whs_checkpoints/New_Snapshots_ct2mr --name "ep12_new_bz16_semi_entropy_50_cps"\
#    --model test --direction AtoB --num_slice 130 --phase test --dataset unalignwhs_test \
#    --log_dir whs_logs/test_logs/ct2mr --epoch '90' --gpu_ids 1
#
#
#python test.py --dataroot ../../datasets/2017-MM-WHS/data --checkpoints_dir whs_checkpoints/New_Snapshots_mr2ct --name "ep12_new_bz16_semi_entropy_50_cps"\
#    --model test --direction BtoA --num_slice 256 --phase test --dataset unalignwhs_test \
#    --log_dir whs_logs/test_logs/mr2ct --epoch '95' --gpu_ids 1
##
#
#  python test.py --dataroot ../../datasets/2017-MM-WHS/data --checkpoints_dir whs_checkpoints/New_Snapshots_ct2mr --name "ep12_new_bz16_semi_entropy_50_cps"\
#    --model test --direction AtoB --num_slice 130 --phase test --dataset unalignwhs_test \
#    --log_dir whs_logs/test_logs/ct2mr --epoch '75' --gpu_ids 1

#  python test.py --dataroot ../../datasets/2017-MM-WHS/data --checkpoints_dir whs_checkpoints/New_Snapshots_ct2mr --name "ep12_new_bz16_semi_entropy_50_cps"\
#    --model test --direction AtoB --num_slice 130 --phase test --dataset unalignwhs_test \
#    --log_dir whs_logs/test_logs/ct2mr --epoch '70' --gpu_ids 1


#  python test.py --dataroot ../../datasets/2017-MM-WHS/data --checkpoints_dir whs_checkpoints/New_Snapshots_mr2ct --name "ep12_new_bz16_semi_cps"\
#    --model test --direction BtoA --num_slice 256 --phase test --dataset unalignwhs_test \
#    --log_dir whs_logs/test_logs/mr2ct --epoch 'best'

#  python test.py --dataroot ../../datasets/2017-MM-WHS/data --checkpoints_dir whs_checkpoints/New_Snapshots_mr2ct --name "ep12_new_bz16_semi_cps"\
#    --model test --direction BtoA --num_slice 256 --phase test --dataset unalignwhs_test \
#    --log_dir whs_logs/test_logs/mr2ct --epoch '95'

#  python test.py --dataroot ../../datasets/abdominal --checkpoints_dir abdo_checkpoints/New_Snapshots_t22ct --name "ep100_new_bz16_semi_cps"\
#    --model test --direction BtoA --case_slice ${case_slice_0[*]} --phase test --dataset unalignabdo_test \
#    --log_dir abdo_log/test_logs/t22ct --epoch 'best' --log_dir abdo_logs/test_logs/t22ct --gpu_ids 1
#
#
#  python test.py --dataroot ../../datasets/abdominal --checkpoints_dir abdo_checkpoints/New_Snapshots_ct2t2 --name "ep50_new_bz16_semi_cps"\
#    --model test --direction AtoB --case_slice ${case_slice_1[*]} --phase test --dataset unalignabdo_test \
#    --log_dir abdo_log/test_logs/ct2t2 --epoch 'best' --log_dir abdo_logs/test_logs/ct2t2 --gpu_ids 1
#
#
#  python test.py --dataroot ../../datasets/abdominal --checkpoints_dir abdo_checkpoints/New_Snapshots_t22ct --name "ep100_new_bz16_semi_entropy_120_cps"\
#    --model test --direction BtoA --case_slice ${case_slice_0[*]} --phase test --dataset unalignabdo_test \
#    --log_dir abdo_log/test_logs/t22ct --epoch 'best' --log_dir abdo_logs/test_logs/t22ct --gpu_ids 1
#
#
#  python test.py --dataroot ../../datasets/abdominal --checkpoints_dir abdo_checkpoints/New_Snapshots_ct2t2 --name "ep50_new_bz16_semi_entropy_120_cps"\
#    --model test --direction AtoB --case_slice ${case_slice_1[*]} --phase test --dataset unalignabdo_test \
#    --log_dir abdo_log/test_logs/ct2t2 --epoch 'best' --log_dir abdo_logs/test_logs/ct2t2 --gpu_ids 1

##precision
#python test.py --dataroot ../../datasets/brats --checkpoints_dir brats_checkpoints/New_Snapshots_FLAIR --name "adaptation"\
#  --model test --direction t22flair --num_slice 155 --phase test --dataset unalignbrats_test --target t2 \
#  --log_dir brats_log/test_logs/flair --epoch '5' --precision 1
#
#python test.py --dataroot ../../datasets/brats --checkpoints_dir brats_checkpoints/New_Snapshots_FLAIR --name "adaptation"\
#  --model test --direction t22t1ce --num_slice 155 --phase test --dataset unalignbrats_test --target t1ce \
#  --log_dir brats_log/test_logs/flair --epoch 'best' --precision 1
#
#
#  python test.py --dataroot ../../datasets/abdominal --checkpoints_dir abdo_checkpoints/New_Snapshots_ct2t2 --name "ep50_new_bz16_semi_entropy_120_cps"\
#    --model test --direction AtoB --case_slice ${case_slice_1[*]} --phase test --dataset unalignabdo_test \
#    --log_dir abdo_log/test_logs/ct2t2 --epoch '100' --log_dir abdo_logs/test_logs/ct2t2 --gpu_ids 1 --precision 1
#
#
#  python test.py --dataroot ../../datasets/abdominal --checkpoints_dir abdo_checkpoints/New_Snapshots_t22ct --name "ep100_new_bz16_semi_entropy_120_cps"\
#    --model test --direction BtoA --case_slice ${case_slice_1[*]} --phase test --dataset unalignabdo_test \
#    --log_dir abdo_log/test_logs/t22ct --epoch 'best' --log_dir abdo_logs/test_logs/t22ct --gpu_ids 1 --precision 1


#  python test.py --dataroot ../../datasets/2017-MM-WHS/data --checkpoints_dir whs_checkpoints/New_Snapshots_mr2ct --name "ep12_new_bz16_semi_cps"\
#    --model test --direction BtoA --num_slice 256 --phase test --dataset unalignwhs_test \
#    --log_dir whs_logs/test_logs/mr2ct --epoch 'best' --gpu_ids 1 --precision 1


  python test.py --dataroot ../../datasets/2017-MM-WHS/data --checkpoints_dir whs_checkpoints/New_Snapshots_ct2mr --name "ep12_new_bz16_semi_entropy_50_cps"\
    --model test --direction AtoB --num_slice 130 --phase test --dataset unalignwhs_test \
    --log_dir whs_logs/test_logs/ct2mr --epoch 'best' --gpu_ids 1 --precision 1
