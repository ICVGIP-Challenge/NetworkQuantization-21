# Do not change the order of models for evaluation
#!/bin/bash
for model in resnet18 mobilenet_v2 shufflenet_v2_x1_0
do
    echo Testing $model model...
    python main.py \
    --model $model \
    --dataset imagenet \
    --test_bs 128 
done