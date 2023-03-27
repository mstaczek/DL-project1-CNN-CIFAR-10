#!/bin/bash 

# This script runs all the tests for the hyperparameters
# This include both hyperparameters and augmentation techniques
# Note: Random MixUp needs a modified version of scripts, located in the folder "random_mixup_scripts"


# parameters order:
# python tests_script.py model repetitions epochs lr batch_size momentum l2_regularisation regularisation_dropout aug_name aug_param
# baseline: 
# python tests_script.py resnet18_pretrained 1 10 0.003 16 0.0 0.0 0.0 None 0.0


echo ""
echo "Starting learning rate"
echo ""
python tests_script.py resnet18_pretrained 1 10 0.01 16 0.0 0.0 0.0 None 0.0
python tests_script.py resnet18_pretrained 1 10 0.001 16 0.0 0.0 0.0 None 0.0
python tests_script.py resnet18_pretrained 1 10 0.0003 16 0.0 0.0 0.0 None 0.0

echo ""
echo "Starting batch size"
echo ""
python tests_script.py resnet18_pretrained 1 10 0.003 2 0.0 0.0 0.0 None 0.0
python tests_script.py resnet18_pretrained 1 10 0.003 128 0.0 0.0 0.0 None 0.0
python tests_script.py resnet18_pretrained 1 10 0.003 1024 0.0 0.0 0.0 None 0.0

echo ""
echo "Starting momentum"
echo ""
python tests_script.py resnet18_pretrained 1 10 0.003 16 0.2 0.0 0.0 None 0.0
python tests_script.py resnet18_pretrained 1 10 0.003 16 0.5 0.0 0.0 None 0.0
python tests_script.py resnet18_pretrained 1 10 0.003 16 0.9 0.0 0.0 None 0.0

echo ""
echo "Starting l2 regularisation"
echo ""
python tests_script.py resnet18_pretrained 1 10 0.003 16 0.0 0.001 0.0 None 0.0
python tests_script.py resnet18_pretrained 1 10 0.003 16 0.0 0.03 0.0 None 0.0
python tests_script.py resnet18_pretrained 1 10 0.003 16 0.0 0.5 0.0 None 0.0


echo ""
echo "Starting regularisation dropout"
echo ""
python tests_script.py resnet18_pretrained 1 10 0.003 16 0.0 0.0 0.2 None 0.0
python tests_script.py resnet18_pretrained 1 10 0.003 16 0.0 0.0 0.4 None 0.0
python tests_script.py resnet18_pretrained 1 10 0.003 16 0.0 0.0 0.8 None 0.0

echo ""
echo "Starting random rotation"
echo ""
python tests_script.py resnet18_pretrained 1 10 0.003 16 0.0 0.0 0.0 random_rotation 30
python tests_script.py resnet18_pretrained 1 10 0.003 16 0.0 0.0 0.0 random_rotation 90
python tests_script.py resnet18_pretrained 1 10 0.003 16 0.0 0.0 0.0 random_rotation 180

echo ""
echo "Starting random crop"
echo ""
python tests_script.py resnet18_pretrained 1 10 0.003 16 0.0 0.0 0.0 random_crop 28
python tests_script.py resnet18_pretrained 1 10 0.003 16 0.0 0.0 0.0 random_crop 16
python tests_script.py resnet18_pretrained 1 10 0.003 16 0.0 0.0 0.0 random_crop 8

echo ""
echo "Starting color jitter"
echo ""
python tests_script.py resnet18_pretrained 1 10 0.003 16 0.0 0.0 0.0 color_jitter 0.1
python tests_script.py resnet18_pretrained 1 10 0.003 16 0.0 0.0 0.0 color_jitter 0.5
python tests_script.py resnet18_pretrained 1 10 0.003 16 0.0 0.0 0.0 color_jitter 1

# echo ""
# echo "Starting random mix up"
# echo ""
# python tests_script.py resnet18_pretrained 1 10 0.003 32 0.0 0.0 0.0 random_mix_up 0.2
# python tests_script.py resnet18_pretrained 1 10 0.003 32 0.0 0.0 0.0 random_mix_up 0.5
# python tests_script.py resnet18_pretrained 1 10 0.003 32 0.0 0.0 0.0 random_mix_up 0.8