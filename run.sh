#!/bin/bash
#SBATCH --output=res.txt
#SBATCH --mem=240000

python multigpu_train.py --gpu_list=0,1,2,3 --input_size=512 --batch_size_per_gpu=64 --checkpoint_path=tmp/east_icdar2015_resnet_v1_50_rbox/ \
--text_scale=512 --training_data_path=Data/cropped_img_train/ --geometry=RBOX --learning_rate=0.00011 --num_readers=24 \
--pretrained_model_path=resnet_v1_50.ckpt
