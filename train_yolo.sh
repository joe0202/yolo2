#!/bin/sh
data_path="/home/qiaolinjun/qljproject/data/trainig_data/test2"
anno_path="/home/qiaolinjun/qljproject/data/trainig_data/test1"
#data_path="/home/qiaolinjun/qljproject/data/copy/im"
#anno_path="/home/qiaolinjun/qljproject/data/copy/an"
#python ./flow --model cfg/tiny-yolo-voc.cfg --train --dataset "/home/qiaolinjun/qljproject/data/VOCdevkit/VOC2007/JPEGImages" --annotation "/home/qiaolinjun/qljproject/data/VOCdevkit/VOC2007/Annotations" --gpu 0.3 --load -1  --lr 0.0001 --save -16000
#data_path="/home/qiaolinjun/qljproject/data/data_training/test2"
#anno_path="/home/qiaolinjun/qljproject/data/data_training/test1"
python ./flow --model cfg/yoloV2-dac.cfg --train --dataset "${data_path}" --annotation "${anno_path}" --gpu 0.6   --lr 0.000001 --save -16000 --load 19000
