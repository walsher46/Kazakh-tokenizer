# Morphology-Aware Segmentation and Tokenization (The Kazakh Case)

This repository provides datasets, preprocessing scripts, and reproducibility
materials for the study **“Morphology-Aware Segmentation and Tokenization for
Turkic Languages: The Kazakh Case”**.

The project addresses morphology-aware tokenization for the Kazakh language using
a segmented corpus derived from the **Complete Set of Endings (CSE)** morphological
model. In addition, it explores improvements to the **FEMSeg-CRF neural segmentation
model** by incorporating Kazakh vowel–consonant harmony features and training on
large-scale CSE-based datasets.

---

## 1. Datasets Overview

### 1.1 Intrinsic Evaluation Datasets

The following datasets are used for intrinsic evaluation:

**Kazakh Raw Sentence Corpus (284,707 sentences)**  
- **Source:** large-scale web crawling of Kazakh-language websites  
- **Preprocessing:** automated deduplication and text cleaning  
- **Usage:** training and evaluation of statistical tokenizers and FEMSeg_kaz_v3  

**CSE-Generated Kazakh Wordform Corpus (2,329,377 wordforms)**  
- **Source:** automatic generation using the CSE morphological model  
- **Usage:** training of FEMSeg_kaz_v2  

### 1.2 Data Availability and Access

Due to storage and source constraints, the complete datasets are not redistributed
directly in this repository. However, the repository provides:

- detailed data access instructions;
- corpus format specifications;
- representative data samples.

These materials are sufficient to reproduce the reported experiments using
independently collected Kazakh text data.

### 1.3 Data Formats

- **Raw corpus:** UTF-8 encoded text files, one sentence per line  
- **CSE-segmented corpus:** UTF-8 text files with morpheme boundaries marked using `@@`  
- **Wordform corpus:** UTF-8 text files, one wordform per line  

---

## 2. Software Environment and Execution

The experiments were primarily conducted using **Python** and executed in
Google Colab; however, the scripts are environment-independent and can be run
locally with appropriate dependencies.

### 2.1 Software Requirements

Typical software environment:

- Python 3.10
- PyTorch
- pytorch-crf
- SentencePiece
- Transformers
- Sentence-Transformers

Exact versions are provided in `requirements.txt`.

### 2.2 Running the Experiments

The repository contains Python scripts (`.py`) implementing the core algorithms
and Jupyter notebooks (`.ipynb`) that orchestrate these scripts to reproduce the
experimental pipelines step by step.

Paths to datasets and output directories are specified explicitly in the scripts
and notebooks. All programs include inline comments explaining the processing
steps, inputs, and outputs.

Separate instructions for extrinsic evaluation experiments are provided in the
corresponding subdirectories.

---

## 3. Reproducibility

Full reproducibility is provided for the main neural model contribution
(**FEMSeg_kaz_v3**) via executable Jupyter notebooks that coordinate the underlying
Python scripts. Other components in the repository represent supporting research
artifacts developed at different stages of the project.

---

## 4. Licensing

### Software

All source code and preprocessing scripts in this repository are released under
the **MIT License** and are provided for research and reproducibility purposes.

### Data

The datasets used in this project are distributed under the
**Creative Commons Attribution 4.0 International (CC BY 4.0)** license.







