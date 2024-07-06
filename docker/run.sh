#!/bin/bash

docker run -it --rm --gpus all \
-v /home/lucasmc/Documents/ufrgs/datasets/FairFace/val:/val_imgs \
-v /home/lucasmc/Documents/ufrgs/vlms_bias_explore:/app \
-w /app \
lucasmc/clip /bin/bash
