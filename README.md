# Graphormer LRGB Evaluation

This project provides a minimal setup to evaluate Hugging Face Graphormer models on the LRGB dataset (e.g., PCQM4Mv2) using torch-geometric and transformers.

## Requirements
- Python 3.8+
- torch
- torch-geometric
- transformers

Install dependencies:
```bash
pip install torch torch-geometric transformers
```

## Usage
Run the evaluation script:
```bash
python eval_graphormer_lrgb.py
```

- The script loads the LRGB dataset and a Graphormer model from Hugging Face.
- It runs a simple evaluation loop and reports accuracy if labels are present.

## Files
- `eval_graphormer_lrgb.py`: Main evaluation script.
- `.github/copilot-instructions.md`: Copilot customization instructions.

## Customization
- Edit the script to change the dataset, model checkpoint, or evaluation metrics as needed.

## References
- [Graphormer (Hugging Face)](https://huggingface.co/docs/transformers/en/model_doc/graphormer)
- [LRGBDataset (torch-geometric)](https://pytorch-geometric.readthedocs.io/en/2.5.1/generated/torch_geometric.datasets.LRGBDataset.html)
