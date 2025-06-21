import argparse
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, mean_absolute_error, roc_auc_score
from graphormer.models.graphormer import GraphormerModel
from graphormer.data.dataset import GraphormerDataset, BatchedDataDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Graphormer')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset spec (e.g., ogbg-molhiv)')
    parser.add_argument('--dataset-source', type=str, default='ogb', choices=['ogb', 'pyg', 'dgl'], help='Dataset source')
    parser.add_argument('--split', type=str, default='test', choices=['test', 'valid', 'val'], help='Dataset split to evaluate')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint (.pt)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--num-classes', type=int, required=True, help='Number of output classes')
    parser.add_argument('--metric', type=str, default='acc', choices=['acc', 'mae', 'auc'], help='Evaluation metric')
    return parser.parse_args()


def main():
    args = parse_args()
    # Load dataset
    dataset = GraphormerDataset(dataset_spec=args.dataset, dataset_source=args.dataset_source)
    if args.split in ['valid', 'val']:
        eval_set = BatchedDataDataset(dataset.dataset_val)
    else:
        eval_set = BatchedDataDataset(dataset.dataset_test)
    eval_loader = DataLoader(eval_set, batch_size=args.batch_size, shuffle=False, collate_fn=eval_set.collater)

    # Model args
    class ModelArgs:
        pass
    model_args = ModelArgs()
    model_args.num_classes = args.num_classes
    model_args.encoder_embed_dim = 768
    model_args.encoder_layers = 12
    model_args.encoder_attention_heads = 32
    model_args.encoder_ffn_embed_dim = 768
    model_args.dropout = 0.1
    model_args.attention_dropout = 0.1
    model_args.act_dropout = 0.1
    model_args.apply_graphormer_init = True
    model_args.activation_fn = 'gelu'
    model_args.encoder_normalize_before = True
    model_args.share_encoder_input_output_embed = False
    model_args.no_token_positional_embeddings = False
    model_args.pre_layernorm = False
    # Add more as needed

    model = GraphormerModel.build_model(model_args)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    model = model.to(args.device)
    model.eval()

    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in eval_loader:
            batch = {k: v.to(args.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            logits = model(batch)
            if args.metric == 'mae':
                pred = logits[:, 0].detach().cpu().numpy()
            else:
                pred = logits.argmax(dim=-1).detach().cpu().numpy()
            target = batch['y'].detach().cpu().numpy() if 'y' in batch else batch['target'].detach().cpu().numpy()
            y_true.extend(target)
            y_pred.extend(pred)

    if args.metric == 'acc':
        acc = accuracy_score(y_true, y_pred)
        print(f'Accuracy: {acc:.4f}')
    elif args.metric == 'mae':
        mae = mean_absolute_error(y_true, y_pred)
        print(f'MAE: {mae:.4f}')
    elif args.metric == 'auc':
        auc = roc_auc_score(y_true, y_pred)
        print(f'AUC: {auc:.4f}')
    else:
        print('Unsupported metric')

if __name__ == '__main__':
    main() 