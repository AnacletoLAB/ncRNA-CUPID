# ncRNA-CUPID
ncRNA-CUPID is a ncRNA-ncRNA interaction classifier, based on a Transformer architecture.

# Requirements
A CUDA environment, and a minimum VRAM of 8GB is required.
### Dependencies
```
torch>=2.0
numpy
transformers==4.33.0.dev0
datasets==2.14.4
tqdm
```

# Usage
Firstly, download the checkpoint of the foundational RNA Language model (GenerRNA)
#### Directory tree
```
.
├── LICENSE
├── README.md
├── model.pt         # to be downloaded
├── model.py         # define the architecture
├── tokenization.py  # preparete data
├── tokenizer        # BPE tokenizer of the foundational RNA LM
├── example_notebook.py # Example usage of ncRNA cupid for training on your set of ncRNA interaction sequences
```

# License
TO-DO