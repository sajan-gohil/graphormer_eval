# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
from dataclasses import dataclass, field

from tqdm import tqdm
import numpy as np

import pickle as pkl

from torch_scatter import scatter


class DiffusionLoss(nn.Module):
    def forward(self, model, sample, reduce=True):
        if model.training:
            output = model.get_training_output(**sample["net_input"])
        else:
            with torch.no_grad():
                output = model.get_sampling_output(**sample["net_input"])

        persample_loss = output["persample_loss"]
        loss = torch.sum(persample_loss)
        return loss


@dataclass
class OCKDEDataclass:
    n_kde_samples: int = field(
        default=10, metadata={"help": "number of samples per oc system"}
    )
    kernel_func: str = field(
        default="normal", metadata={"help": "kernel function used in KDE"}
    )
    kde_temperature: float = field(
        default=0.1, metadata={"help": "temperature of kernel function"}
    )
    result_save_dir: str = field(
        default="flow_ode_res_save_dir",
        metadata={"help": "directory to save flow ode results"},
    )
    density_calc_z_offset: int = field(
        default=0,
        metadata={
            "help": "z offset, will be multiplied by 0.1 and subtract from the z coordinate of hydrogen atom"
        },
    )


def rmsd_pbc(batched_pos1, batched_pos2, cell, lig_mask, num_moveable):
    delta_pos = (
        (batched_pos2.unsqueeze(2) - batched_pos1.unsqueeze(1)).float()
    )  # B x N1 x N2 x T x 3
    cell = cell.unsqueeze(1).unsqueeze(2).float()  # B x 1 x 1 x 3 x 3
    delta_pos_solve = torch.linalg.solve(
        cell.transpose(-1, -2), delta_pos.transpose(-1, -2)
    ).transpose(-1, -2)
    delta_pos_solve[:, :, :, :, 0] %= 1.0
    delta_pos_solve[:, :, :, :, 0] %= 1.0
    delta_pos_solve[:, :, :, :, 1] %= 1.0
    delta_pos_solve[:, :, :, :, 1] %= 1.0
    delta_pos_solve[delta_pos_solve > 0.5] -= 1.0
    min_delta_pos = torch.matmul(delta_pos_solve, cell)
    rmsds = torch.sqrt(
        torch.sum(torch.sum(min_delta_pos**2, dim=-1) * lig_mask, dim=-1)
        / num_moveable.unsqueeze(1).unsqueeze(2)
    )
    return rmsds


class OCKDE(DiffusionLoss):
    def __init__(self, cfg: OCKDEDataclass):
        super().__init__()
        self.n_kde_samples = cfg.n_kde_samples
        self.kernel_func = cfg.kernel_func
        self.kde_temperature = cfg.kde_temperature
        self.result_save_dir = cfg.result_save_dir

    def calc_kde_probs_from_rmsd(self, rmsd):
        all_probs = (
            1.0
            / np.sqrt(2.0 * 3.1415926)
            * torch.exp(-(rmsd**2) / self.kde_temperature)
        )  # B x N1 x N2
        probs = torch.mean(all_probs, axis=-1) / self.kde_temperature  # B x N1
        probs /= torch.sum(probs, axis=-1, keepdim=True) + 1e-32
        return probs

    def forward(self, model, sample, reduce=True):
        if model.training:
            return super().forward(model, sample, reduce)
        else:
            batched_data = sample["net_input"]["batched_data"]
            batched_data["init_pos"] = batched_data["pos"].clone()
            if "all_poses" not in batched_data and "index_i" not in batched_data:
                return super().forward(model, sample, reduce)
            else:
                if "all_poses" not in batched_data:
                    batched_data["all_poses"] = batched_data["pos"].clone().unsqueeze(1)
                num_moveable = batched_data["num_moveable"]
                all_poses = batched_data["all_poses"]
                all_outputs = []
                ori_pos = batched_data["pos"].clone()
                for _ in tqdm(range(self.n_kde_samples)):
                    with torch.no_grad():
                        batched_data["pos"] = ori_pos.clone()
                        output = model.get_sampling_output(batched_data)
                        pred_pos = output["pred_pos"]  # B x T x 3
                    all_outputs.append(pred_pos.clone().unsqueeze(-3))
                all_outputs = torch.cat(all_outputs, axis=-3)  # B x N1 x T x 3
                device = str(num_moveable.device).replace(":", "_")
                with open(f"{self.result_save_dir}/{device}_pos.out", "ab") as out_file:
                    sids = batched_data["sid"]
                    atoms = batched_data["x"][:, :, 0]
                    tags = batched_data["tags"]
                    pkl.dump((sids, atoms, tags, all_outputs, all_poses), out_file)

                atoms = batched_data["x"][:, :, 0]
                mask = (~atoms.eq(0)).unsqueeze(1).unsqueeze(2).float()

                rmsds = torch.sqrt(
                    torch.sum(
                        torch.sum(
                            (all_outputs.unsqueeze(1) - all_outputs.unsqueeze(2)) ** 2,
                            dim=-1,
                        )
                        * mask,
                        dim=-1,
                    )
                    / num_moveable.unsqueeze(1).unsqueeze(2)
                )

                rmsds_ref = torch.sqrt(
                    torch.sum(
                        torch.sum(
                            (all_outputs.unsqueeze(2) - all_poses.unsqueeze(1)) ** 2,
                            dim=-1,
                        )
                        * mask,
                        dim=-1,
                    )
                    / num_moveable.unsqueeze(1).unsqueeze(2)
                )

                self_probs = self.calc_kde_probs_from_rmsd(rmsds)  # B x N1
                ref_probs = self.calc_kde_probs_from_rmsd(rmsds_ref)  # B x N1
                kl_div = torch.sum(
                    self_probs * torch.log(self_probs / (ref_probs + 1e-32)), axis=1
                ) / (torch.sum(self_probs, dim=1) + 1e-32)
                kl_div_sum = torch.sum(kl_div)
                return kl_div_sum


class FlowODE(DiffusionLoss):
    def __init__(self, cfg: OCKDEDataclass):
        super().__init__()
        self.n_kde_samples = cfg.n_kde_samples
        self.kernel_func = cfg.kernel_func
        self.kde_temperature = cfg.kde_temperature
        self.result_save_dir = cfg.result_save_dir

    def calc_kde_probs_from_rmsd(self, rmsd):
        all_probs = (
            1.0
            / np.sqrt(2.0 * 3.1415926)
            * torch.exp(-(rmsd**2) / self.kde_temperature)
        )  # B x N1 x N2
        probs = torch.mean(all_probs, axis=-1) / self.kde_temperature  # B x N1
        probs /= torch.sum(probs, axis=-1, keepdim=True) + 1e-32
        return probs

    def forward(self, model, sample, reduce=True):
        if model.training:
            return super().forward(model, sample, reduce)
        else:
            batched_data = sample["net_input"]["batched_data"]
            if "all_poses" not in batched_data:
                return super().forward(model, sample, reduce)
            else:
                num_moveable = batched_data["num_moveable"]
                all_poses = batched_data["all_poses"]
                all_outputs = []
                ori_pos = batched_data["pos"].clone()
                for _ in tqdm(range(self.n_kde_samples)):
                    with torch.no_grad():
                        batched_data["pos"] = ori_pos.clone()
                        output = model.get_sampling_output(batched_data)
                        pred_pos = output["pred_pos"]  # B x T x 3
                    all_outputs.append(pred_pos.clone().unsqueeze(-3))
                all_outputs = torch.cat(all_outputs, axis=-3)  # B x N1 x T x 3
                device = str(num_moveable.device).replace(":", "_")
                with open(f"{self.result_save_dir}/{device}_pos.out", "ab") as out_file:
                    sids = batched_data["sid"]
                    atoms = batched_data["x"][:, :, 0]
                    tags = batched_data["tags"]
                    pkl.dump((sids, atoms, tags, all_outputs, all_poses), out_file)

                atoms = batched_data["x"][:, :, 0]
                mask = (~atoms.eq(0)).unsqueeze(1).unsqueeze(2).float()

                rmsds = torch.sqrt(
                    torch.sum(
                        torch.sum(
                            (all_outputs.unsqueeze(1) - all_outputs.unsqueeze(2)) ** 2,
                            dim=-1,
                        )
                        * mask,
                        dim=-1,
                    )
                    / num_moveable.unsqueeze(1).unsqueeze(2)
                )

                rmsds_ref = torch.sqrt(
                    torch.sum(
                        torch.sum(
                            (all_outputs.unsqueeze(2) - all_poses.unsqueeze(1)) ** 2,
                            dim=-1,
                        )
                        * mask,
                        dim=-1,
                    )
                    / num_moveable.unsqueeze(1).unsqueeze(2)
                )

                self_probs = self.calc_kde_probs_from_rmsd(rmsds)  # B x N1
                ref_probs = self.calc_kde_probs_from_rmsd(rmsds_ref)  # B x N1
                kl_div = torch.sum(
                    self_probs * torch.log(self_probs / (ref_probs + 1e-32)), axis=1
                ) / (torch.sum(self_probs, dim=1) + 1e-32)
                kl_div_sum = torch.sum(kl_div)
                return kl_div_sum


class FlowODECalcDensity(DiffusionLoss):
    def __init__(self, cfg: OCKDEDataclass):
        super().__init__()
        self.result_save_dir = cfg.result_save_dir
        self.density_calc_z_offset = cfg.density_calc_z_offset

    def forward(self, model, sample, reduce=True):
        if model.training:
            return super().forward(model, sample, reduce)
        else:
            batched_data = sample["net_input"]["batched_data"]
            batched_data["init_pos"] = batched_data["pos"].clone()
            with torch.no_grad():
                output = model.get_sampling_output(batched_data)
                pred_pos = output["pred_pos"]
                device = str(pred_pos.device).replace(":", "_")
                with open(
                    f"{self.result_save_dir}/{device}_density.out", "ab"
                ) as out_file:
                    sids = batched_data["sid"]
                    atoms = batched_data["x"][:, :, 0]
                    tags = batched_data["tags"]
                    pkl.dump((sids, atoms, tags, pred_pos), out_file)

            return torch.tensor(0.0, device=pred_pos.device)
