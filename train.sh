export CUDA_VISIBLE_DEVICES=2
source activate python35
source set-cudnn 6
python xmlgencsv_train.py
python keras_retinanet/bin/train.py --backbone='resnet101' --weights='./weights/ResNet-101-model.keras.h5' --batch-size=1  --snapshot-path='./snapshots_resnet101_crab_snail/' --image-min-side=2048 --image-max-side=2048 --tensorboard-dir="./logs/crab_snail/" csv iscas_frcnn_train.csv iscas_frcnn_class_train.csv
