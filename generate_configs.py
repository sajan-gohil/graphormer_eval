# parser.add_argument("--edge_type", type=str, default="multi_hop", help="Type of edge encoding (multi_hop, single_hop, etc.)")
# parser.add_argument("--enable_spatial_encoder", action="store_true", help="Enable spatial encoder")
# parser.add_argument("--enable_diffusion", action="store_true", help="Enable diffusion")
# parser.add_argument("--optimize_diffuser", action="store_true", help="Optimize diffuser  # NOT USED. DEPRECATED")
# parser.add_argument("--tensor_parallel", action="store_true", help="Enable tensor parallelism on 2 GPUs (requires >=2 GPUs)")
# parser.add_argument("--experiment_dir", type=str, default="./experiments", help="Directory to save experiment results")
# parser.add_argument("--name", type=str, default="graphormer_experiment", help="Name of the experiment")
# parser.add_argument("--reconstruction_scale", type=float, default=0.0, help="How much to weigh diffusion reconstruction loss")
# parser.add_argument("--structure_scale", type=float, default=0.0, help="How much to weigh diffusion reconstruction loss")
# parser.add_argument("--aug_loss_scale", type=float, default=1, help="How much to weigh augmentation correction's reconstruction loss")
# parser.add_argument("--onscreen_logs", action="store_true", help="print logs on screen instead of log files in experiment dir")
# parser.add_argument("--batch_size", type=int, default=512, help="number of graphs in a batch")
# parser.add_argument("--diffusion_type", type=str, default="x0", help='Type of diffusion predictor ["x0", "delta", "noise_pred"]')
# parser.add_argument("--detached_denoiser", action="store_true", help="Detach embedding before passing to diffusion module to separate denoiser training")
# parser.add_argument("--pretrained_weights", type=str, default=None, help="path to checkpoint pt file")
# parser.add_argument("--diffusion_steps", type=int, default=50, help="Number of diffusion steps for the model")
# parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for data loading")
# parser.add_argument("--dataset_name", type=str, default="pcqm4mv2", help="Name of the dataset to use")
# parser.add_argument("--create_subgraph", action="store_true", help="Create subgraphs from given large graph")
# parser.add_argument("--num_denoiser_layers", type=int, default=3, help="Number of layers in the denoiser: n down, n-1 up + 1 final projection")
# parser.add_argument("--use_linear_denoiser", action="store_true", help="[Deprecated with denoisers.py] Use linear layers in the denoiser")
# parser.add_argument("--denoiser_type", type=str, default="gat", help="Denoiser layer type. Accepted: gat, linear, mha")

# parser.add_argument("--optimize_only_diffuser", action="store_true", help="Optimize only the diffuser model")
# parser.add_argument("--augment_edges", action="store_true", help="Remove/add dummy edges and calculate separate loss")
# parser.add_argument("--gnn_only", action="store_true", help="Instead of diffusion, treat denoiser as gnn")
# parser.add_argument("--remove_attn_bias", action="store_true", help="Remove attention bias module altogether")
# parser.add_argument("--enable_layerwise_diffusion", action="store_true", help="Perform diffusion after each attention step")
# parser.add_argument("--freeze_pretrained_encoder", type=str, default=None, help="Freeze the pretrained encoder and set weights from given path")


param_tree = {
    "dataset_name": ["cora"], # "cora", "citeseer", "pubmed"
    "remove_attn_bias": {
        # True: {},
        False: {
            "edge_type": ["single_hop"],
            "enable_spatial_encoder": [False],
        },
    },
    "enable_diffusion": {
        # True: {
        #     "denoiser_type": ["gat", "mha"], # "linear",
        #     "gnn_only": {
        #         True: {
        #             "diffusion_type": ["x0"]
        #         },
        #         False: {
        #             "diffusion_type": ["x0", "noise_pred", "noise_pred_single"],  # "delta",
        #             # "reconstruction_scale": [0.0, 1.0],
        #             # "structure_scale": [0.0, 1.0],
        #             # "diffusion_steps": [50, 100],
        #             # "num_denoiser_layers": [2, 3, 4],
        #             "optimize_only_diffuser": {
        #                 True: {
        #                     "pretrained_weights": "placeholder"
        #                 },
        #                 False: {
        #                     # "detached_denoiser": [True, False],
        #                 }
        #             },
        #             # "augment_edges": {
        #             #     True: {"aug_loss_scale": [0.0, 0.5, 1.0],},
        #             #     False: {}
        #             # }
        #         }
        #     }
        # },
        False: {}
    }
}


from copy import deepcopy

def expand_tree(tree, prefix=None):
    """
    Recursively expand parameter tree into fully specified configs.
    Each config includes all top-level params that apply.
    """
    if prefix is None:
        prefix = {}

    # If nothing left to expand, this is a complete config
    if not tree:
        return [prefix]

    # Grab the first key to expand
    key, values = next(iter(tree.items()))
    rest = {k: v for k, v in tree.items() if k != key}
    configs = []

    if isinstance(values, list):
        for v in values:
            new_prefix = deepcopy(prefix)
            new_prefix[key] = v
            configs.extend(expand_tree(rest, new_prefix))

    elif isinstance(values, dict):
        for choice, subtree in values.items():
            new_prefix = deepcopy(prefix)
            new_prefix[key] = choice
            # Merge subtree with the rest so we don’t lose other top-levels
            merged = deepcopy(rest)
            merged.update(subtree)
            configs.extend(expand_tree(merged, new_prefix))

    else:  # direct assignment
        new_prefix = deepcopy(prefix)
        new_prefix[key] = values
        configs.extend(expand_tree(rest, new_prefix))

    return configs


# Generate all configs
configs = expand_tree(param_tree)


print(configs)
########################################

# Add name. "cora_"+ diff if diffusion, gnn of gnn_only else base + <denoiser_type> + <diffusion type> + ("no_edge" if single_hop + no_spatial if no spatial_enoder) or no_bias if remove_attn_bias + "rec_<rec_scale>_struc_<struc_scale>"
for config in configs:
    name = config["dataset_name"] + "_"

    if config.get("enable_diffusion", False):
        if config["gnn_only"]:
            name += "gnn_"
        else:
            name += "diff_"
        if config["denoiser_type"] is not None:
            name += config["denoiser_type"] + "_"
        if config.get("diffusion_type", None) is not None:
            name += config["diffusion_type"] + "_"
        # name += "no_rec_"
        name += f"rec_{config.get('reconstruction_scale', 0.0)}_" if config.get("reconstruction_scale", 0.0) > 0 else ""
        name += f"struc_{config.get('structure_scale', 0.0)}_" if config.get("structure_scale", 0.0) > 0 else ""
    else:
        name += "base_"
    
    name += "only_" if config.get("optimize_only_diffuser", False) else ""

    if config.get("remove_attn_bias", False):
        name += "no_bias_"
    else:
        if config.get("edge_type", "multi_hop") == "single_hop":
                name += "no_edge_"
        if not config.get("enable_spatial_encoder", True):
            name += "no_spatial_"
    config["name"] = name.strip("_").replace(".", "")
    if config.get("optimize_only_diffuser", False):
        if "no_edge_no_spatial" in config["name"]:
            config["pretrained_weights"] = "experiments/cora_base_no_edge_no_spatial_2025-09-21_10-57-26/training_checkpoints/best_model_1542.pt"
        elif "no_edge" in config["name"]:
            config["pretrained_weights"] = "experiments/cora_base_no_edge_2025-09-21_11-27-18/training_checkpoints/best_model_1648.pt"
        elif "no_spatial" in config["name"]:
            config["pretrained_weights"] = "experiments/cora_base_no_spatial_2025-09-21_18-18-35/training_checkpoints/best_model_1131.pt"
        elif "no_bias" in config["name"]:
            raise Exception("NO VALID MODEL")
            pass
        else:
            config["pretrained_weights"] = "experiments/cora_base_2025-09-21_19-00-31/training_checkpoints/best_model_1131.pt"


# Save configs to file
import json
with open("configs.json", "w") as f:
    json.dump(configs, f, indent=4)
print(f"Generated {len(configs)} configurations and saved to configs.json")