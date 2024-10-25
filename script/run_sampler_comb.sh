OPTIMIZER=$1
DEVICES=$2
CUDA_VISIBLE_DEVICES=$DEVICES python ./src/pytorch/run.py \
    -d "../data/ISCAS/c17.bench" -l\
    --optimizer "mse"\
    --lr 15e0 \
    --problem_type "BLIF" \
    --circuit_type "comb" \
    --num_steps 1 \
    --batch_size 1000 \
    --wandb_entity "ucb-hcrl" \
    --wandb_project "gdsampler" \
    --wandb_group "debug" \
    --wandb_tags "seed=0"