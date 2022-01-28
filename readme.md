# Install

```
conda create -n faket
conda activate faket
conda install pytorch cudatoolkit=11.3 -c pytorch
conda install pip
pip insatll -r requirements.txt
```

# Reproducing the results

Open the `main.ipynb` and run all the cells.
This assumes you are on Ubuntu machine with available GPU, reasonably fast CPU, installed Nvidia drivers, and you are running the jupyter notebook from within the conda environment created in the Section Install.