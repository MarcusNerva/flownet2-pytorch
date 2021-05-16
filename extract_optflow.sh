#!/bin/bash
# >>> conda initialize >>>

# -----------Please paste your conda setting here-----------

# <<< conda initialize <<<

conda activate flownet2

#Please modify following arguments according to your operating environment
#name of your video dataset
dataset_name="MSVD"
#path of your video dataset
video_dir="/home/hanhuaye/PythonProject/YouTubeClips"
#videos' extension type
ext="avi"
#extract optical flow for every 5 frames
stride=5
#how many optical flow videos would you like to see?
thresh_hold=10
#There are 7 varients of flownet2, namely, [FlowNet2-CSS-ft-sd, FlowNet2-C, FlowNet2, FlowNet2-CSS,
#FlowNet2-SD, FlowNet2-CS, FlowNet2-S]. FlowNet2 is recommended.
model_name="FlowNet2"

python extract_flow.py \
    --dataset_name $dataset_name \
    --video_dir $video_dir \
    --model_name $model_name \
    --stride $stride \
    --ext $ext

python show_optflow.py \
    --dataset_name $dataset_name\
    --thresh_hold $thresh_hold

