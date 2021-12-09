#!/usr/bin/env bash

cd ..

python trainval.py --dataset KITTIRoad2D --mode train --batch_size 6 --epochs 200
python trainval.py --dataset KITTIRoad2D --mode test --batch_size 1
python trainval.py --dataset KITTIRoad3D --mode train --batch_size 8 --epochs 200
python trainval.py --dataset KITTIRoad3D --mode test --batch_size 1