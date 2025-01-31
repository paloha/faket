![git_header](https://gitlab.com/-/project/30876804/uploads/c8260febe31a7632a807fe464ad95f57/git_header.jpg)

# FakET: Simulating Cryo-Electron Tomograms with Neural Style Transfer


This project proposes FakET (pronounced `fake E.T.`), a novel method for simulating the forward operator of a cryo-transmission electron microscope to generate synthetic micrographs or tilt-series. It was created, among other reasons, to generate training data for deep neural networks to solve tasks such as localization and classification of biological particles. It is based on additive noise and neural style transfer. Instead of a calibration protocol, it only requires unlabelled reference data. It does not need to be retrained in order to be used on your data. FakET is capable of simulating large tilt-series, which are common in experimental environments. For example, it can generate a $61\times3500\times3500$ tilt-series on a single *NVIDIA A100 40GB SXM4* GPU in less than 10 minutes. It therefore has the potential to save experts countless hours of manual work in labeling their data sets in pursuit of obtaining annotated data for training their models in a supervised fashion.

**Preprint:** The method and its evaluation is described in this paper [arXiv:2304.02011](https://arxiv.org/abs/2304.02011). :page_facing_up:

**News:** The article was accepted in [Cell Structure](https://www.cell.com/structure/home) journal. :tada:

**Outlook:** We are working on further validation. :hammer:

----

## System Requirements & Install

In order to use FakET, you will ideally need a computer with a GPU and installed NVIDIA & CUDA drivers. However, also a CPU-only simulation is feasible. We have developed and tested the simulator on an Nvidia HGX A100 supercomputer running Ubuntu 20.04 OS, nevertheless, also a standard laptop should suffice to run the simulations, naturally, the computation will be slower. We handle the package dependencies using Conda environments that can be easily re-created using the provided `environment.yml` files. More info about the package versions is available within each `environment.yml` file. 

```bash
# Clone the repository
git clone https://gitlab.com/deepet/faket.git
cd faket

# Create the desired CONDA environment
conda env create -f <ENVIRONMENT_FILENAME>
conda activate <ENVIRONMENT_NAME>
```

## Simulations

### CPU-only simulations

**Use the following Conda environment:** \
ENVIRONMENT_FILENAME: `environment-cpu.yml` \
ENVIRONMENT_NAME: `faketCPU`

Assuming the existence of `tiltseries_content.mrc`, `tiltseries_style.mrc`, and `tiltseries_noisy.mrc`, all of size e.g. `61 x 1024 x 1024`, while having e.g. 32 CPU cores available, the simulation can be started by calling the command bellow. The simulated tilt-series will be stored in the file `tiltseries_output.mrc` specified by `--output` argument. *Heads-up:* the code for creating the `tiltseries_noisy.mrc` is available in the `faket/noisy.py` with an example in the `main.ipynb`, however, the CLI interface is not yet done.

Command example:

```bash
python3 -m faket.style_transfer.cli some_folder/tiltseries_content.mrc some_folder/tiltseries_style.mrc --init some_folder/tiltseries_noisy.mrc --output some_folder/tiltseries_output.mrc --devices cpu --ncpus 32 --random-seed 0 --min-scale 1024 --end-scale 1024 --seq_start 0 --seq_end 61 --style-weights 1.0 --content-weight 1.0 --tv-weight 0 --iterations 1 --initial-iterations 1 --save-every 2 --step-size 0.15 --avg-decay 0.99 --style-scale-fac 1.0 --pooling max --content_layers 8 --content_layers_weights 100 --model_weights pretrained
```

### GPU simulations

**Use the following Conda environment:** \
ENVIRONMENT_FILENAME: `environment-gpu.yml` \
ENVIRONMENT_NAME: `faketGPU`

Here we assume the same as in the CPU-only simulations, except now we also assume having a GPU available. Use the same command as before, just prepend it with `CUDA_VISIBLE_DEVICES=0` or any other GPU ID that you have currently available, and use `--devices cuda:0`. In this case `--ncpus` is irrelevant.

Command example:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m faket.style_transfer.cli some_folder/tiltseries_content.mrc some_folder/tiltseries_style.mrc --init some_folder/tiltseries_noisy.mrc --output some_folder/tiltseries_output.mrc --devices cuda:0 --random-seed 0 --min-scale 1024 --end-scale 1024 --seq_start 0 --seq_end 61 --style-weights 1.0 --content-weight 1.0 --tv-weight 0 --iterations 1 --initial-iterations 1 --save-every 2 --step-size 0.15 --avg-decay 0.99 --style-scale-fac 1.0 --pooling max --content_layers 8 --content_layers_weights 100 --model_weights pretrained
```

----

## Viewing the results from the paper in more detail :eyes:

In case you are interested in a deeper dive into the results without actually going through the trouble of reproducing them yourself, visit the `reproduce/archived_results` folder where we stored the final results of presented methods per class or per task along with the details of performance of each of the selected best epochs in csv files. Moreover, in the additional experiment folders you can also find logs of training, segmentation, and clustering, as well as full evaluation of the best epoch on the test tomogram. Most importantly, all the figures are stored in the `figures.ipynb` file and can be just viewed without running anything (but since the file is large, you will need to download it and view it locally).

----

## Reproducing the results from the paper :rocket:

### Prepare the environment

**Use the following Conda environment:** \
ENVIRONMENT_FILENAME: `environment-reproduce.yml` \
ENVIRONMENT_NAME: `faketREPRODUCE`

The original experiments were done on a headless server with the following specifications: \
**OS Version:** Ubuntu 20.04.3 LTS \
**NVIDIA Driver Versions:** 510.47.03 and 510.85.02 \
**CUDA Version:** 11.6 \
**GPU Model:** 8x NVIDIA A100-SXM4 40GB

### Prepare the data

1. Create `data/shrec2021_extended_dataset` folder.
1. Download `shrec2021_original_groundtruth.zip` and `shrec2021_full_dataset.zip` files into it from [here](https://dataverse.nl/dataset.xhtml?persistentId=doi:10.34894/XRTJMA).
1. Extract the `shrec2021_original_groundtruth.zip` first.
1. Rename all `model_x/groundtruth.mrc` to `model_x/groundtruth_unbinned.mrc`.
1. Extract the `shrec2021_full_dataset.zip` into the same directory.

The original_groundtruth zip contains 3 additional files per tomogram:

* `grandmodel.mrc` - Unbinned ground truth (1024, 1024, 1024), be aware of the name collision with `grandmodel.mrc` from full_dataset.
* `noisefree_projections.mrc` - Projections of grandmodel embedded in ice (61, 1024, 1024).
* `projections.mrc` - Projections with noise before CTF scaling (61, 1024, 1024).

For even more info go to [shrec webpage](https://www.shrec.net/cryo-et/) or [DOI 10.2312/3dor.20211307](https://diglib.eg.org/bitstream/handle/10.2312/3dor20211307/005-017.pdf).

### Follow the steps in `main.ipynb`

To reproduce all the results presented in the paper, activate the conda environment created in Section Install, run JupyterLab (which is already available in the Conda environment) and follow the instructions within the `main.ipynb`. Please note, that creating the data modalities is going to be reasonably fast, but it will take a lot of time to run all the evaluation experiments using the DeepFinder. For submission of jobs we have used the SLURM submission system. Some of the steps can be run directly within the `main.ipynb` notebook itself, however, many are submitted to SLURM via SBATCH scripts, therefore it is beneficial if you have SLURM installed and you know how to use it. It is not necessary to submit jobs via SLURM to reproduce our results, however, our code assumes it. The exact `.sbatch` commands we have issued to produce the results in the paper are stored in the `reproduce` folder.

### Follow the steps in `figures.ipynb`

After all the steps (creating projections, NST, computing reconstructions, training DeepFinder, segmentation, clustering, evaluation) from `main.ipynb` were executed successfully, it is time to visualize the data and the results. Follow the instructions in `figures.ipynb` to produce all the figures and data for the tables presented in the paper.
