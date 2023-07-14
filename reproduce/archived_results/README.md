# Results folder

This folder contains best epoch results for each of the models. We did not include results and checkpoints for every epoch as this would be a waste of storage.


**WARNING: Mind the new naming convention!**

Before the article submission, but after running all the experiments, we changed the naming of the methods to make our experiments and aims easier to understand. Therefore, in this folder, `<oldname>` is mapped to `<NEWNAME>` used in the paper according to the map bellow. E.g. folder `exp_baseline` contains data on experiments in `best_epochs_BENCHMARK`, etc.

```python
naming = {
    'baseline': 'BENCHMARK',  # data reconstructed from SHREC projections
    'noiseless': 'NOISELESS',  # projections obtained by negating the result of radon transform
    'gauss': 'BASELINE',  # noiseless projections + simple gaussian noise
    'noisy': 'NOISY',  # noiseless projections + tilt-aware gaussian noise
    'styled': 'FAKET',  # NOISY projections style transferred with NST
    'baseline3': 'BENCHMARK-3',  # training only using 3 BENCHMARK tomos
    'finetune': 'FINETUNED-3',  # training with FAKET for 50 epochs and then fine-tuned using 3 BENCHMARK tomos
    'rstyled': 'FAKET_RANDOM',  # ablation study with randomly initialized NST network (instead of using pretrained weights on ImageNet)
}
```
