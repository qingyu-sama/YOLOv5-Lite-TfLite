#!/bin/bash

python export_tflite.py --weights runs/train/exp18/weights/best.pt --img-size 320 320 --int8 --full-integer --data /home/qingyu/jupyter/swim/dataset/data.yaml --num-calibration-images 20 --device cpu
vela runs/train/exp18/weights/best_saved_model/best_full_integer_quant.tflite --accelerator-config ethos-u65-256 --system-config Ethos_U65_High_End