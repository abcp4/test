#!/usr/bin/env bash
# GRID SEARCH
# ALPHA SEARCH
( python3 main.py -gpu 0 -miniception D_ordinal -lr 1e-6 -n_blocks 4 -alpha 1
python3 main.py -gpu 1 -miniception D_ordinal -lr 1e-6 -n_blocks 4 -alpha 2
python3 main.py -gpu 2 -miniception D_ordinal -lr 1e-6 -n_blocks 4 -alpha 3
python3 main.py -gpu 0 -miniception D_ordinal -lr 1e-6 -n_blocks 4 -alpha 4
python3 main.py -gpu 0 -miniception D_ordinal -lr 1e-6 -n_blocks 4 -alpha 5
python3 main.py -gpu 0 -miniception D_ordinal -lr 1e-6 -n_blocks 4 -alpha 6
python3 main.py -gpu 0 -miniception D_ordinal -lr 1e-6 -n_blocks 4 -alpha 7
python3 main.py -gpu 0 -miniception D_ordinal -lr 1e-6 -n_blocks 4 -alpha 8
python3 main.py -gpu 0 -miniception D_ordinal -lr 1e-6 -n_blocks 4 -alpha 9
python3 main.py -gpu 0 -miniception D_ordinal -lr 1e-6 -n_blocks 4 -alpha 10

# N_BLOCKS SEARCH
python3 main.py -gpu 0 -miniception D_ordinal -lr 1e-6 -n_blocks 5 -alpha 1
python3 main.py -gpu 0 -miniception D_ordinal -lr 1e-6 -n_blocks 5 -alpha 2
python3 main.py -gpu 0 -miniception D_ordinal -lr 1e-6 -n_blocks 5 -alpha 3
python3 main.py -gpu 0 -miniception D_ordinal -lr 1e-6 -n_blocks 5 -alpha 4
python3 main.py -gpu 0 -miniception D_ordinal -lr 1e-6 -n_blocks 5 -alpha 5
python3 main.py -gpu 0 -miniception D_ordinal -lr 1e-6 -n_blocks 5 -alpha 6
python3 main.py -gpu 0 -miniception D_ordinal -lr 1e-6 -n_blocks 5 -alpha 7
python3 main.py -gpu 0 -miniception D_ordinal -lr 1e-6 -n_blocks 5 -alpha 8
python3 main.py -gpu 0 -miniception D_ordinal -lr 1e-6 -n_blocks 5 -alpha 9
python3 main.py -gpu 0 -miniception D_ordinal -lr 1e-6 -n_blocks 5 -alpha 10

# LR SEARCH
python3 main.py -gpu 0 -miniception D_ordinal -lr 1e-4 -n_blocks 4 -alpha 1
python3 main.py -gpu 0 -miniception D_ordinal -lr 5e-4 -n_blocks 4 -alpha 1
python3 main.py -gpu 0 -miniception D_ordinal -lr 1e-5 -n_blocks 4 -alpha 1
python3 main.py -gpu 0 -miniception D_ordinal -lr 5e-5 -n_blocks 4 -alpha 1
python3 main.py -gpu 0 -miniception D_ordinal -lr 5e-6 -n_blocks 4 -alpha 1
python3 main.py -gpu 0 -miniception D_ordinal -lr 1e-7 -n_blocks 4 -alpha 1 ) &

(python3 main.py -gpu 1 -miniception D_ordinal -lr 1e-4 -n_blocks 5 -alpha 1
python3 main.py -gpu 1 -miniception D_ordinal -lr 5e-4 -n_blocks 5 -alpha 1
python3 main.py -gpu 1 -miniception D_ordinal -lr 1e-5 -n_blocks 5 -alpha 1
python3 main.py -gpu 1 -miniception D_ordinal -lr 5e-5 -n_blocks 5 -alpha 1
python3 main.py -gpu 1 -miniception D_ordinal -lr 5e-6 -n_blocks 5 -alpha 1
python3 main.py -gpu 1 -miniception D_ordinal -lr 1e-7 -n_blocks 5 -alpha 1

# HOLDOUT
python3 main.py -gpu 1 -holdout D_ordinal -lr 1e-6 -n_blocks 4 -alpha 1
python3 main.py -gpu 1 -holdout D_ordinal -lr 1e-6 -n_blocks 4 -alpha 2
python3 main.py -gpu 1 -holdout D_ordinal -lr 1e-6 -n_blocks 4 -alpha 3
python3 main.py -gpu 1 -holdout D_ordinal -lr 1e-6 -n_blocks 4 -alpha 4
python3 main.py -gpu 1 -holdout D_ordinal -lr 1e-6 -n_blocks 4 -alpha 5
python3 main.py -gpu 1 -holdout D_ordinal -lr 1e-6 -n_blocks 4 -alpha 6
python3 main.py -gpu 1 -holdout D_ordinal -lr 1e-6 -n_blocks 4 -alpha 7
python3 main.py -gpu 1 -holdout D_ordinal -lr 1e-6 -n_blocks 4 -alpha 8
python3 main.py -gpu 1 -holdout D_ordinal -lr 1e-6 -n_blocks 4 -alpha 9
python3 main.py -gpu 1 -holdout D_ordinal -lr 1e-6 -n_blocks 4 -alpha 10

# HOLDOUT R17
python3 main.py -gpu 1 -holdout D_ordinal -config miniception_R17 -alpha 4 -lr 1e-6
python3 main.py -gpu 1 -holdout D_ordinal -config miniception_R17 -alpha 4 -lr 1e-6
python3 main.py -gpu 1 -holdout D_ordinal -config miniception_R17 -alpha 4 -lr 1e-6
python3 main.py -gpu 1 -holdout D_ordinal -config miniception_R17 -alpha 4 -lr 1e-6
python3 main.py -gpu 1 -holdout D_ordinal -config miniception_R17 -alpha 4 -lr 1e-6

python3 main.py -gpu 1 -holdout D_ordinal -aug true -config miniception_R17 -alpha 4 -lr 1e-6 -aug_images 50
python3 main.py -gpu 1 -holdout D_ordinal -aug true -config miniception_R17 -alpha 4 -lr 1e-6 -aug_images 100
python3 main.py -gpu 1 -holdout D_ordinal -aug true -config miniception_R17 -alpha 4 -lr 1e-6 -aug_images 200
python3 main.py -gpu 1 -holdout D_ordinal -aug true -config miniception_R17 -alpha 4 -lr 1e-6 -aug_images 300
python3 main.py -gpu 1 -holdout D_ordinal -aug true -config miniception_R17 -alpha 4 -lr 1e-6 -aug_images 400
python3 main.py -gpu 1 -holdout D_ordinal -aug true -config miniception_R17 -alpha 4 -lr 1e-6 -aug_images 500 ) &

# HOLDOUT R33
(python3 main.py -gpu 2 -holdout D_ordinal -config miniception_R33 -alpha 4 -lr 1e-6
python3 main.py -gpu 2 -holdout D_ordinal -config miniception_R33 -alpha 4 -lr 1e-6
python3 main.py -gpu 2 -holdout D_ordinal -config miniception_R33 -alpha 4 -lr 1e-6
python3 main.py -gpu 2 -holdout D_ordinal -config miniception_R33 -alpha 4 -lr 1e-6
python3 main.py -gpu 2 -holdout D_ordinal -config miniception_R33 -alpha 4 -lr 1e-6

python3 main.py -gpu 2 -holdout D_ordinal -aug true -config miniception_R33 -alpha 4 -lr 1e-6 -aug_images 50
python3 main.py -gpu 2 -holdout D_ordinal -aug true -config miniception_R33 -alpha 4 -lr 1e-6 -aug_images 100
python3 main.py -gpu 2 -holdout D_ordinal -aug true -config miniception_R33 -alpha 4 -lr 1e-6 -aug_images 200
python3 main.py -gpu 2 -holdout D_ordinal -aug true -config miniception_R33 -alpha 4 -lr 1e-6 -aug_images 300
python3 main.py -gpu 2 -holdout D_ordinal -aug true -config miniception_R33 -alpha 4 -lr 1e-6 -aug_images 400
python3 main.py -gpu 2 -holdout D_ordinal -aug true -config miniception_R33 -alpha 4 -lr 1e-6 -aug_images 500

# HOLDOUT R49
python3 main.py -gpu 2 -holdout D_ordinal -config miniception_R49 -alpha 4 -lr 1e-6
python3 main.py -gpu 2 -holdout D_ordinal -config miniception_R49 -alpha 4 -lr 1e-6
python3 main.py -gpu 2 -holdout D_ordinal -config miniception_R49 -alpha 4 -lr 1e-6
python3 main.py -gpu 2 -holdout D_ordinal -config miniception_R49 -alpha 4 -lr 1e-6
python3 main.py -gpu 2 -holdout D_ordinal -config miniception_R49 -alpha 4 -lr 1e-6

python3 main.py -gpu 2 -holdout D_ordinal -aug true -config miniception_R49 -alpha 4 -lr 1e-6 -aug_images 50
python3 main.py -gpu 2 -holdout D_ordinal -aug true -config miniception_R49 -alpha 4 -lr 1e-6 -aug_images 100
python3 main.py -gpu 2 -holdout D_ordinal -aug true -config miniception_R49 -alpha 4 -lr 1e-6 -aug_images 200
python3 main.py -gpu 2 -holdout D_ordinal -aug true -config miniception_R49 -alpha 4 -lr 1e-6 -aug_images 300
python3 main.py -gpu 2 -holdout D_ordinal -aug true -config miniception_R49 -alpha 4 -lr 1e-6 -aug_images 400
python3 main.py -gpu 2 -holdout D_ordinal -aug true -config miniception_R49 -alpha 4 -lr 1e-6 -aug_images 500 )
