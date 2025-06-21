import argparse
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from graphormer.models.graphormer import GraphormerModel
from graphormer.data.dataset import GraphormerDataset, BatchedDataDataset, TargetDataset
from graphormer.criterions.multiclass_cross_entropy import GraphPredictionMulticlassCrossEntropy


def parse_args():
    parser = argparse.ArgumentParser(description='Train Graphormer')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset spec (e.g., ogbg-molhiv)')
    parser.add_argument('--dataset-source', type=str, default='ogb', choices=['ogb', 'pyg', 'dgl'], help='Dataset source')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--output-dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--num-classes', type=int, required=True, help='Number of output classes')
    # Add more model hyperparameters as needed
    parser.add_argument('--num-workers', type=int, default=1, help='number of data loading workers')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    dataset = GraphormerDataset(dataset_spec=args.dataset, dataset_source=args.dataset_source)
    train_loader = DataLoader(
        dataset.dataset_train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=BatchedDataDataset(dataset.dataset_train).collater
    )
    val_loader = DataLoader(
        dataset.dataset_val, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=BatchedDataDataset(dataset.dataset_val).collater
    )

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
    model_args.num_atoms = 512 * 9
    model_args.num_edges = 512 * 3
    model_args.num_in_degree = 512
    model_args.num_out_degree = 512
    model_args.num_spatial = 512
    model_args.num_edge_dis = 128
    model_args.edge_type = 'multi_hop'
    model_args.multi_hop_max_dist = 5
    model_args.max_nodes = 128

    model = GraphormerModel.build_model(model_args)
    model = model.to(args.device)

    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = {k: v.to(args.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            optimizer.zero_grad()
            logits = model(batch)
            targets = batch['y']
            loss = criterion(logits, targets.long().squeeze(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}: Train Loss = {avg_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(args.device) if torch.is_tensor(v) else v for k, v in batch.items()}
                logits = model(batch)
                targets = batch['y']
                loss = criterion(logits, targets.long().squeeze(-1))
                val_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                total += targets.size(0)
                correct += (predicted == targets.squeeze(-1)).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        acc = correct / total if total > 0 else 0
        print(f"Epoch {epoch}: Val Loss = {avg_val_loss:.4f}, Acc = {acc:.4f}")

        # Save checkpoint
        torch.save(model.state_dict(), os.path.join(args.output_dir, f'graphormer_epoch{epoch}.pt'))

if __name__ == '__main__':
    main() 