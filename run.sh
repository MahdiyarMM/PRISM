# python main.py --dataset WaTerBirds --mitigation train --epoch 1 --wandb "First_waterbird_RN50_base" --seed 1 --CLIP_model RN50 --lr 0.05 --num_samples 1000 --num_bases 7
# python main.py --dataset celebA --mitigation train --epoch 1 --wandb "First_celebA" --seed 1 --lr 0.1 --num_samples 1000 
# python main.py --dataset WaTerBirds --mitigation train --epoch 1 --seed 0  --num_samples 300 --lr 0.001  --CLIP_model RN50
# python main.py --dataset celeba --mitigation train --epoch 1 --wandb celebA_Reg_cos --seed 0 --num_samples 100 --lr 0.01 --reg_type cos --reg_lambda 0.01
# python main.py --dataset celeba --mitigation train --epoch 1 --wandb celebA_Reg_gram --seed 0 --num_samples 100 --lr 0 --reg_type gram --reg_lambda 0.01 --init_weight I
# python main.py --dataset celeba --mitigation train --epoch 1 --wandb celebA_Reg_orth --seed 0 --num_samples 100 --lr 0.01 --reg_type orth --reg_lambda 0.01
# python main.py --dataset celeba --mitigation train --epoch 1 --wandb celebA_Reg_both --seed 0 --num_samples 100 --lr 0.01 --reg_type both --reg_lambda 0.01

# python main.py --dataset celeba  --mitigation train --seed 0 --num_samples 50 --lr 0.1  --init_weight I --wandb celebA_I --reg_type orth --reg_lambda 0.5
# python main.py --dataset celeba --mitigation train --epoch 1 --wandb waterbird_orth --seed 0 --num_samples 100 --lr 0.01 --reg_type orth --reg_lambda 0.5

# python main.py --dataset celeba --mitigation train --epoch 1 --wandb celebA_SCENE --seed 0 --num_samples 100 --lr 0.01
# python main.py --dataset celeba --mitigation train --epoch 1 --wandb celebA_1_base --seed 0 --num_samples 100 --lr 0.01 --num_bases 1
# python main.py --dataset celeba --mitigation train --epoch 1 --wandb celebA_2_base --seed 0 --num_samples 100 --lr 0.01 --num_bases 2
# python main.py --dataset celeba --mitigation train --epoch 1 --wandb celebA_3_base --seed 0 --num_samples 100 --lr 0.01 --num_bases 3
# python main.py --dataset celeba --mitigation train --epoch 1 --wandb celebA_4_base --seed 0 --num_samples 100 --lr 0.01 --num_bases 4
# python main.py --dataset celeba --mitigation train --epoch 1 --wandb celebA_5_base --seed 0 --num_samples 100 --lr 0.01 --num_bases 5

# python main.py --dataset celeba --mitigation train --epoch 1 --wandb celebA_SCENE_b0 --seed 0 --num_samples 100 --lr 0.01
# python main.py --dataset celeba --mitigation train --epoch 1 --wandb celebA_SCENE_b1 --seed 0 --num_samples 100 --lr 0.01 --num_bases 1
# python main.py --dataset celeba --mitigation train --epoch 1 --wandb celebA_SCENE_b2 --seed 0 --num_samples 100 --lr 0.01 --num_bases 2
# python main.py --dataset celeba --mitigation train --epoch 1 --wandb celebA_SCENE_b3 --seed 0 --num_samples 100 --lr 0.01 --num_bases 3
# python main.py --dataset celeba --mitigation train --epoch 1 --wandb celebA_SCENE_b4 --seed 0 --num_samples 100 --lr 0.01 --num_bases 4
# python main.py --dataset celeba --mitigation train --epoch 1 --wandb celebA_SCENE_b5 --seed 0 --num_samples 100 --lr 0.01 --num_bases 5


# python main.py --dataset celeba --mitigation train --epoch 1 --wandb celebA_SCENE_s10 --seed 0 --num_samples 10 --lr 0.01
# python main.py --dataset celeba --mitigation train --epoch 1 --wandb celebA_SCENE_s20 --seed 0 --num_samples 20 --lr 0.01
# python main.py --dataset celeba --mitigation train --epoch 1 --wandb celebA_SCENE_s30 --seed 0 --num_samples 30 --lr 0.01
# python main.py --dataset celeba --mitigation train --epoch 1 --wandb celebA_SCENE_s40 --seed 0 --num_samples 40 --lr 0.01
# python main.py --dataset celeba --mitigation train --epoch 1 --wandb celebA_SCENE_s50 --seed 0 --num_samples 50 --lr 0.01
# python main.py --dataset celeba --mitigation train --epoch 1 --wandb celebA_SCENE_s60 --seed 0 --num_samples 60 --lr 0.01
# python main.py --dataset celeba --mitigation train --epoch 1 --wandb celebA_SCENE_s70 --seed 0 --num_samples 70 --lr 0.01
# python main.py --dataset celeba --mitigation train --epoch 1 --wandb celebA_SCENE_s80 --seed 0 --num_samples 80 --lr 0.01
# python main.py --dataset celeba --mitigation train --epoch 1 --wandb celebA_SCENE_s90 --seed 0 --num_samples 90 --lr 0.01
# python main.py --dataset celeba --mitigation train --epoch 1 --wandb celebA_SCENE_s100 --seed 0 --num_samples 100 --lr 0.01

# python main.py --dataset celeba --mitigation train --epoch 1 --wandb celebA_SCENE_s70 --seed 0 --num_samples 70 --lr 0.01

# python main.py --dataset celeba --mitigation train --epoch 1 --wandb celebA_SCENE_s70_seed0 --seed 0 --num_samples 70 --lr 0.01
# python main.py --dataset celeba --mitigation train --epoch 1 --wandb celebA_SCENE_s70_seed1 --seed 1 --num_samples 70 --lr 0.01
# python main.py --dataset celeba --mitigation train --epoch 1 --wandb celebA_SCENE_s70_seed2 --seed 2 --num_samples 70 --lr 0.01
# python main.py --dataset celeba --mitigation train --epoch 1 --wandb celebA_SCENE_s70_seed3 --seed 3 --num_samples 70 --lr 0.01
# python main.py --dataset celeba --mitigation train --epoch 1 --wandb celebA_SCENE_s70_seed4 --seed 4 --num_samples 70 --lr 0.01
# python main.py --dataset celeba --mitigation train --epoch 1 --wandb celebA_SCENE_s70_seed5 --seed 5 --num_samples 70 --lr 0.01
# python main.py --dataset celeba --mitigation train --epoch 1 --wandb celebA_SCENE_s70_seed6 --seed 6 --num_samples 70 --lr 0.01
# python main.py --dataset celeba --mitigation train --epoch 1 --wandb celebA_SCENE_s70_seed7 --seed 7 --num_samples 70 --lr 0.01
# python main.py --dataset celeba --mitigation train --epoch 1 --wandb celebA_SCENE_s70_seed8 --seed 8 --num_samples 70 --lr 0.01
# python main.py --dataset celeba --mitigation train --epoch 1 --wandb celebA_SCENE_s70_seed9 --seed 9 --num_samples 70 --lr 0.01
# python main.py --dataset celeba --mitigation train --epoch 1 --wandb celebA_SCENE_s70_seed10 --seed 10 --num_samples 70 --lr 0.01


# python main.py --dataset celeba --mitigation train --epoch 1 --wandb celebA_SCENE_s100_seed0 --seed 0 --num_samples 100 --lr 0.01
# python main.py --dataset celeba --mitigation train --epoch 1 --wandb celebA_SCENE_s100_seed1 --seed 1 --num_samples 100 --lr 0.01
# python main.py --dataset celeba --mitigation train --epoch 1 --wandb celebA_SCENE_s100_seed2 --seed 2 --num_samples 100 --lr 0.01
# python main.py --dataset celeba --mitigation train --epoch 1 --wandb celebA_SCENE_s100_seed3 --seed 3 --num_samples 100 --lr 0.01
# python main.py --dataset celeba --mitigation train --epoch 1 --wandb celebA_SCENE_s100_seed4 --seed 4 --num_samples 100 --lr 0.01
# python main.py --dataset celeba --mitigation train --epoch 1 --wandb celebA_SCENE_s100_seed5 --seed 5 --num_samples 100 --lr 0.01
# python main.py --dataset celeba --mitigation train --epoch 1 --wandb celebA_SCENE_s100_seed6 --seed 6 --num_samples 100 --lr 0.01
# python main.py --dataset celeba --mitigation train --epoch 1 --wandb celebA_SCENE_s100_seed7 --seed 7 --num_samples 100 --lr 0.01
# python main.py --dataset celeba --mitigation train --epoch 1 --wandb celebA_SCENE_s100_seed8 --seed 8 --num_samples 100 --lr 0.01
# python main.py --dataset celeba --mitigation train --epoch 1 --wandb celebA_SCENE_s100_seed9 --seed 9 --num_samples 100 --lr 0.01
python main.py --dataset celeba --mitigation train --epoch 1 --wandb celebA_SCENE_s120_seed10 --seed 10 --num_samples 120 --lr 0.01
python main.py --dataset celeba --mitigation train --epoch 1 --wandb celebA_SCENE_s140_seed10 --seed 10 --num_samples 140 --lr 0.01
python main.py --dataset celeba --mitigation train --epoch 1 --wandb celebA_SCENE_s160_seed10 --seed 10 --num_samples 160 --lr 0.01