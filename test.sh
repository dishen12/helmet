export CUDA_VISIBLE_DEVICES=0
source activate python35
source set-cudnn 6
python xmlgencsv_test.py
#python keras_retinanet/bin/convert_model.py './snapshots_resnet101_crab_snail/resnet101_csv_07.h5' './inferenceshots_resnet101_snail/resnet101_csv_07.h5'
python keras_retinanet/bin/evaluate.py csv iscas_frcnn_test_crab.csv iscas_frcnn_class_test_crab.csv  './inferenceshots_resnet50--crab/resnet50_csv_29.h5' 



