#!/bin/bash

# This script repeats the baseline experiment 5 times

# parameters order:
# python tests_script.py model repetitions epochs lr

echo ""
echo "Starting training ResNet18 pretrained"
echo ""
python tests_script.py simple_nn 5 25 0.003

echo ""
echo "Starting training ResNet18 pretrained"
echo ""
python tests_script.py simple_cnn 5 25 0.003

echo ""
echo "Starting training ResNet18 pretrained"
echo ""
python tests_script.py resnet18_pretrained 5 25 0.003

echo ""
echo "Starting training ResNet18 not pretrained"
echo ""
python tests_script.py resnet18_not_pretrained 5 25 0.003