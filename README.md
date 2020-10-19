To install the environment with the dependecies:
```
conda env create -f environment.yml
```

To make the environment lightweighted we decided not to include the
`jupyter-notebook` package.
Therefore to run the example provided in this folder you have two
option:
1. Install the `notebook` package in the **pyfferaph** environment via
```
conda activate pyfferaph
conda install -c conda-forge notebook
```
and then you launch the notebook as usual,

**OR**

2.
