import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from ogb.lsc import PCQM4Mv2Dataset, PCQM4MEvaluator

from graphormer_hf.modeling_graphormer import GraphormerForGraphClassification
from graphormer_hf.configuration_graphormer import GraphormerConfig
from graphormer_hf.collating_graphormer import GraphormerDataCollator

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load preprocessed pyg dataset
pyg_data = torch.load("pyg_dataset_ogb.pt")
split_dict = pyg_data.get_idx_split()
valid_idx = split_dict['valid']
testdev_idx = split_dict['test-dev']

# for i in range(len(pyg_data)):
#     pyg_data[i].num_nodes = pyg_data[i].x.shape[0]

# Subsets
valid_dataset = Subset(pyg_data, valid_idx)
testdev_dataset = Subset(pyg_data, testdev_idx)

# DataLoader
BATCH_SIZE = 1024
collator = GraphormerDataCollator(on_the_fly_processing=True)

valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator)
testdev_loader = DataLoader(testdev_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator)

# Evaluator
evaluator = PCQM4MEvaluator()

def evaluate(model, dataloader, name="valid"):
    model.eval()
    y_pred, y_true = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {name}"):
            for k in batch:
                batch[k] = batch[k].to(device)
            labels = batch['labels']
            outputs = model(**batch)
            y_pred.append(outputs[1].view(-1).cpu())
            y_true.append(labels.view(-1).cpu())

    y_pred = torch.cat(y_pred, dim=0)
    y_true = torch.cat(y_true, dim=0)

    input_dict = {"y_true": y_true.numpy(), "y_pred": y_pred.numpy()}
    result = evaluator.eval(input_dict)
    return result['mae']

# 1. Load fine-tuned model
# print("\n==> Evaluating Fine-Tuned Model (best_model.pt)")
# fine_tuned_config = GraphormerConfig(
#     num_hidden_layers=12,
#     embedding_dim=768,
#     ffn_embedding_dim=768,
#     num_attention_heads=32,
#     dropout=0.0,
#     attention_dropout=0.1,
#     activation_dropout=0.1,
#     num_classes=1,
# )
# fine_tuned_model = GraphormerForGraphClassification(fine_tuned_config)
# fine_tuned_model.load_state_dict(torch.load("best_model.pt", map_location=device))
# fine_tuned_model.to(device)

# valid_mae = evaluate(fine_tuned_model, valid_loader, name="validation")
# testdev_mae = evaluate(fine_tuned_model, testdev_loader, name="test-dev")
# print(f"Fine-tuned Model - Validation MAE: {valid_mae:.6f}")
# print(f"Fine-tuned Model - Test-Dev MAE:   {testdev_mae:.6f}")

# 2. Load Hugging Face pretrained model
print("\n==> Evaluating HuggingFace Pretrained Model")
hf_model = GraphormerForGraphClassification.from_pretrained("clefourrier/graphormer-base-pcqm4mv2")
hf_model.to(device)

valid_mae_hf = evaluate(hf_model, valid_loader, name="validation")
testdev_mae_hf = evaluate(hf_model, testdev_loader, name="test-dev")
print(f"HuggingFace Model - Validation MAE: {valid_mae_hf:.6f}")
print(f"HuggingFace Model - Test-Dev MAE:   {testdev_mae_hf:.6f}")
