# Install

In order to use Faket, you will need a computer with available GPU, reasonably fast CPU, installed NVIDIA drivers and CUDA drivers. We handle the package dependencies using Conda environment which can be easily re-created using the `environment.yml` file. More info about the package versions is available within the `environment.yml` file.  We have tested the environment on a headless server with the following specifications: 
Date: 15.02.2022,
OS Version: Ubuntu 20.04.3 LTS,
NVIDIA Driver Version: 510.47.03,
CUDA Version: 11.6,
GPU Model: 8x NVIDIA A100-SXM4 40GB.

Re-create the Conda environment using:

```
cd faket
conda env create -f environment.yml
conda activate faket
```


# Reproducing the results

## Prepare the data

1. Create `data/shrec2021_extended_dataset` folder.
1. Download [shrec2021_original_groundtruth.zip](https://drive.google.com/file/d/15WLX23h8pnSlm5jC3gpv9UrbC-5ZG_fv) and [shrec2021_full_dataset.zip](https://drive.google.com/file/d/12RlpjhKd2Mi29-uocX9exrt83egwHWlK) files into it. 
1. Extract the `shrec2021_original_groundtruth.zip` first. 
1. Rename all `model_x/groundtruth.mrc` to `model_x/groundtruth_unbinned.mrc`. 
1. Extract the `shrec2021_full_dataset.zip` into the same directory.

The original_groundtruth zip contains 3 additional files per tomogram: 

* `grandmodel.mrc` - Unbinned ground truth (1024, 1024, 1024), be aware of the name collision with `grandmodel.mrc` from full_dataset.
* `noisefree_projections.mrc` - Projections of grandmodel embedded in ice (61, 1024, 1024).
* `projections.mrc` - Projections with noise before CTF scaling (61, 1024, 1024). 

For even more info go to [shrec webpage](https://www.shrec.net/cryo-et/) or [DOI 10.2312/3dor.20211307](https://diglib.eg.org/bitstream/handle/10.2312/3dor20211307/005-017.pdf).

## Run `main.ipynb`

To reproduce all the results presented in the paper, just activate the conda environment created in Section Install and run all the cells within the `main.ipynb` using e.g. JupyterLab (which is already available in the Conda environment).
Please note, that creating the data modalities is going to be reasonably fast, but it will take a lot of time to run all the evaluation experiments using the DeepFinder.

