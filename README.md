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







# Morphology-Aware Segmentation and Tokenization (The Kazakh Case) 
A morphology-aware tokenizer for Kazakh fine-tuned using a segmented corpus derived from the Complete Set of Endings (CSE) morphological model. Using the SentencePiece framework, our tokenizer is fine-tuned on the segmented CSE data to preserve natural morpheme boundaries in Kazakh. The second area is exploring improvements to the FemSeg-CRF segmentation neural network model by incorporating vowel/consonant harmony and training it on a large-volume CSE model-based dataset. 

# 1. Datasets Overview
## 1.1. Intrinsic Estimation Datasets Overview
This part of project uses the following datasets:
1) Kazakh raw sentence corpus (284,707 sentences)
   - Source: large-scale web crawling of Kazakh-language websites
   - Preprocessing: automated deduplication and cleaning
   - Usage: training and evaluation of statistical tokenizers and FEMSeg_kaz_v3

2) CSE-generated Kazakh wordform corpus (2,329,377 wordforms)
   - Source: automatic generation using the CSE morphological model
   - Usage: training FEMSeg_kaz_v2

## 1.2. Data Availability and Access
Due to GitHub storage limitations, only partial versions of the corpora are included in the repository. The complete datasets are not redistributed directly. 
However, detailed instructions for data access, data format specifications, and representative samples are provided.
Researchers interested in accessing the full datasets for academic use may contact the authors.
## 1.3. Data Format
- Raw corpus: UTF-8 text files, one sentence per line.
- CSE-segmented corpus: UTF-8 text files with morpheme boundaries marked by '@@'.
- Wordform corpus: one wordform per line.

  # 2. General instructions for running programs.

Since the programs were run in Google Colab, the description is specific to that environment.

## 2.1	The first cell mounts the Colab directory for the given program. 
For example:
  from google.colab import drive
  drive.mount("/content/drive", force_remount=True)
  curr_dir = "/content/drive/MyDrive/KAZAKH_BPE"
  %cd "$curr_dir"

## 2.2	This directory should store the source files for this program. The program's output is also written to this directory.

## 2.3  The source data required for the program can be determined from the file paths. 
  For example:
  # Paths
    corpus_path = "/content/drive/MyDrive/KAZAKH_BPE/chunk_099.txt"
    spm_model_path = "/content/drive/MyDrive/KAZAKH_BPE/kazakh_bpe.model",

    The first line specifies the path to the source file, and the second line specifies the path to the program module being used.

## 2.4 The program also specifies the import of the necessary packages for running the program. 
  For example:
    import math
    from collections import Counter
    from tqdm import tqdm
    import sentencepiece as spm
    from transformers import MT5Tokenizer, AutoTokenizer

Sometimes, installing a required library is required to import a required package. For example:
    !pip install -q pytorch-crf

    import torch
    import torch.nn as nn
    from torchcrf import CRF
    from torch.utils.data import Dataset, DataLoader

All of this is written in the program text, written in Python.

## 2.5 The program text contains comments describing the program's steps. 
  For example:
  # Paths and preparation of the 10k corpus# -------------
    BASE_DIR = "/content/drive/MyDrive/KAZ_MORPH"
    os.makedirs(BASE_DIR, exist_ok=True)

    SENT_FULL_PATH = os.path.join(BASE_DIR, "kazakh_segmented_corpus_284707.txt")
    SENT_50K_PATH  = os.path.join(BASE_DIR, "kazakh_segmented_corpus_50k.txt")
    CHAR2ID_PATH   = os.path.join(BASE_DIR, "char2id_femseg_v3_50k.json")
    MODEL_DIR      = os.path.join(BASE_DIR, "models_femseg_v3_50k")
    os.makedirs(MODEL_DIR, exist_ok=True)

## 2.5 The program results can be output to a file or directly to the screen.
  For example:
    # Printing results    
    print(f"Actual corpus vocabulary size: {vocab_size}")
    print(f"Total number of tokens: {total_tokens}")
    print(f"Total number of characters before tokenization: {total_chars}")
    print(f"Compression ratio (characters/tokens): {compression_ratio: .4f}")
    print(f"Theoretical Bits per token (log2 vocabulary): {theoretical_bits_per_token: .4f}")
    print(f"Real Bits per token (entropy): {entropy: .4f}")
    print(f"Total information (theoretical): {total_info_theoretical/1e6: .2f} Mbps")
    print(f"Total information (real): {total_info_real/1e6: .2f} Mbps")
или  вывод в файл.

## 2.6  For Extrinsic Estimation, instructions are provided separately for each program in the corresponding directory.

## License and Data Availability

### Software
All source code and preprocessing scripts in this repository are released under the **MIT License** and are provided for research and reproducibility purposes.

### Data

The datasets used in this project are distributed under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license.
