
# Installation

```[bash]
    pip install torch (for appropriate CUDA version)
    pip install numpy
    pip install tqdm
```

# run code

```[bash]
CUDA_VISIBLE_DEVICES=$GPU_ID python src/run.py --num_epochs 100 -v -l --dataset_path "data/flat30/*.cnf"
CUDA_VISIBLE_DEVICES=$GPU_ID python src/run.py --num_epochs 100 -v -l --dataset_path "data/pigeon_hole/*.cnf"
```


