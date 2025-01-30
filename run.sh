# python main.py --dataset WaTerBirds --mitigation train --epoch 1 --wandb "First_waterbird_RN50_base" --seed 1 --CLIP_model RN50 --lr 0.05 --num_samples 1000 --num_bases 7
# python main.py --dataset celebA --mitigation train --epoch 1 --wandb "First_celebA" --seed 1 --lr 0.1 --num_samples 1000 
python main.py --dataset WaTerBirds --mitigation train --epoch 1 --seed 0  --num_samples 300 --lr 0.001  --CLIP_model RN50
