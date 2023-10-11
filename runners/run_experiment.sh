#!/usr/bin/env bash
cd ..


#passa dataset as parameter

#python3 -m cProfile -o program.prof  main.py --dataset_dir "/mnt/sda4/mab-datasets" --parallel_pool_size 15 --datasets $1

python3 main.py --dataset_dir "/users/earoza/earoza/mab-datasets" --parallel_pool_size 15 --datasets $1

