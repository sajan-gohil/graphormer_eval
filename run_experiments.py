import json
import subprocess

CONFIG_FILE = "configs.json"
SCRIPT = "train_graphormer.py"

# Generate the config file if it doesn't exist
try:
    with open(CONFIG_FILE, "r") as f:
        configs = json.load(f)
except FileNotFoundError:
    import generate_configs  # Assumes this creates configs.json
    with open(CONFIG_FILE, "r") as f:
        configs = json.load(f)

for i, config in enumerate(configs):
    args = []
    for k, v in config.items():
        if isinstance(v, bool):
            if v:
                args.append(f"--{k}")
            # skip if False
        else:
            args.extend([f"--{k}", str(v)])
    print(f"Running experiment {i+1}/{len(configs)}: {' '.join(args)}")
    subprocess.run(["python", SCRIPT] + args)