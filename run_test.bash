# Do not change the order of models for evaluation
#!/bin/bash
for model in resnet50_test inception_v3_google_test mobilenet_v3_large_test
do
    echo Testing $model model...
    python main.py \
    --model $model \
    --dataset imagenet \
    --test_bs 128 
done