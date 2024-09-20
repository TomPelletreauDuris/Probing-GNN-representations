Here is the updated `README.md` based on your additional details:

---

# Probing-GNN-Representations

This repository contains various Jupyter notebooks and data related to probing different Graph Neural Networks (GNNs) using a variety of datasets. The focus is on probing models like GCN, GIN, GAT, and R-GCN across multiple datasets, including a **Grid-House** dataset inspired from [this paper](https://arxiv.org/abs/2210.15304), the **AIFB** entities dataset from [torch geometric](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html), **ABIDE (ASD)** and **REST-meta-MDD** (MDD) for brain imaging data coming from the [BrainIB paper](https://ieeexplore.ieee.org/document/10680255) and **ClinTox** for molecular data 

## Project Structure

The repository is organized as follows:

### Folders:

- `.ipynb_checkpoints/`  
  Jupyter notebook checkpoints that are automatically created when you save a notebook.

- `data/`  
  Contains input datasets used for GNN probing.

- `Datasets/`  
  A folder containing datasets and notebooks for generating and exploring data.
  - **Explore generated dataset.ipynb**: A notebook that allows for exploring the generated dataset.

- `models/`  
  Stores the GCN architectures and trained models for various GNN configurations, including GCN, GIN, and R-GCN.

- `results/`  
  Contains output results, such as performance metrics and predictions from the GNN models after experimentation.

- `Testes/`  
  A placeholder or testing folder for experimental files.

### Files:

- `.gitignore`  
  Specifies files and directories that should be ignored by Git, such as large datasets or checkpoint files.

- `.gitattributes`  
  Defines attributes for various files, likely specifying the use of Git LFS or line ending settings.

- `README.md`  
  The file you're currently reading, which provides an overview of the project structure.

- `requirements.txt`  
  Contains a list of Python dependencies required to run the notebooks and experiments in this repository.

### Jupyter Notebooks:

- **[GC] Load trained model.ipynb**  
  Notebook for loading a pre-trained GCN model.

- **[NC] Load trained model.ipynb**  
  Another notebook for loading a different type of trained model.

- **AIFB_probing_GCN.ipynb**  
  Experiments using the AIFB dataset with a GCN model.

- **AIFB_probing_RGCN.ipynb**  
  Experiments using the AIFB dataset with an R-GCN model.

- **ClinTox_probing_GAT.ipynb**  
  Experiments using the ClinTox dataset with a GAT model.

- **ClinTox_probing_GCN.ipynb**  
  Experiments using the ClinTox dataset with a GCN model.

- **ClinTox_probing_GIN.ipynb**  
  Experiments using the ClinTox dataset with a GIN model.

- **FC_probing_GAT.ipynb**  
  Experiments using the ABIDE (ASD) dataset with a GAT model.

- **FC_probing_GCN.ipynb**  
  Experiments using the ABIDE (ASD) dataset with a GCN model.

- **FC_probing_GCN_MDD.ipynb**  
  Experiments using the REST-meta-MDD dataset with a GCN model. 

- **FC_probing_GIN.ipynb**  
  Experiments using the ABIDE (ASD) dataset with a GIN model.

- **Probing_GAT.ipynb**  
  General probing notebook for GAT models.

- **Probing_GCN.ipynb**  
  General probing notebook for GCN models.

- **Probing_GIN.ipynb**  
  General probing notebook for GIN models.

- **Probing_R_GCN.ipynb**  
  General probing notebook for R-GCN models.

### Datasets

- **ABIDE (ASD)**  
  Used in `FC_probing_GAT.ipynb`, `FC_probing_GCN.ipynb`, and `FC_probing_GIN.ipynb`. This dataset pertains to Autism Spectrum Disorder (ASD).

- **REST-meta-MDD**  
  This dataset is used in `FC_probing_GCN_MDD.ipynb` and is related to Major Depressive Disorder (MDD). For more information ob both datasets, refer to the [BrainIB paper](https://ieeexplore.ieee.org/document/10680255).

## How to Use

1. **Install Dependencies:**

   To run the notebooks, ensure that the required Python packages are installed. You can install the dependencies listed in the `requirements.txt` file by running:

   ```bash
   pip install -r requirements.txt
   ```

2. **Run Jupyter Notebooks:**

   After installing the dependencies, you can run the notebooks using Jupyter:

   ```bash
   jupyter notebook
   ```

   Then, open any of the provided `.ipynb` files to begin experimenting with different GNN models and datasets.

## Notes

- The `models/` directory contains GCN architectures and trained models, so make sure you have sufficient space to store them.
- Results generated from the experiments will be stored in the `results/` directory.
- To explore the datasets, check out the **Explore generated dataset.ipynb** in the `Datasets/` folder.
