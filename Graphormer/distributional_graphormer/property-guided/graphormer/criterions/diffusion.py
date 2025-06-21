# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from tqdm import tqdm
import numpy as np
import pickle as pkl


class DiffusionLoss(nn.Module):
    def forward(self, model, sample, reduce=True):
        if model.training:
            output = model.get_training_output(**sample["net_input"])
        else:
            with torch.no_grad():
                output = model.get_sampling_output(**sample["net_input"])

        persample_loss = output["persample_loss"]
        loss = torch.sum(persample_loss)

        if "persample_radius_loss" in output:
            persample_radius_loss = torch.sum(output["persample_radius_loss"])
            radius_loss = torch.sum(persample_radius_loss)
            loss += radius_loss
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
                all_likelihood = []
                all_prior_log_p = []
                all_pred_pos = []
                all_latent_pos = []
                ori_pos = batched_data["pos"].clone()
                for _ in tqdm(range(self.n_kde_samples)):
                    with torch.no_grad():
                        batched_data["pos"] = ori_pos.clone()
                        output = model.get_sampling_output(batched_data)
                        pred_pos = output["pred_pos"]  # B x T x 3
                        all_pred_pos.append(pred_pos.clone().unsqueeze(1))
                        batched_data["pos"] = ori_pos.clone()
                        batched_data["pred_pos"] = pred_pos
                        flow_ode_output = model.get_flow_ode_output(batched_data)
                        all_likelihood.append(
                            flow_ode_output["persample_likelihood"].unsqueeze(1)
                        )
                        all_prior_log_p.append(
                            flow_ode_output["persample_prior_log_p"].unsqueeze(1)
                        )
                        all_latent_pos.append(
                            flow_ode_output["latent_pos"].unsqueeze(1)
                        )
                        device = str(num_moveable.device).replace(":", "_")
                        with open(f"{self.result_save_dir}/{device}.out", "ab") as out_file:
                            sids = batched_data["sid"]
                            pkl.dump(
                                (
                                    sids,
                                    all_likelihood[-1],
                                    all_prior_log_p[-1],
                                    all_pred_pos[-1],
                                    all_latent_pos[-1],
                                ),
                                out_file,
                            )
                        all_outputs.append(pred_pos.clone().unsqueeze(-3))
                all_outputs = torch.cat(all_outputs, axis=-3)  # B x N1 x T x 3
                all_likelihood = torch.cat(all_likelihood, axis=-1)
                all_likelihood_max, _ = torch.max(all_likelihood, axis=1)
                self_probs = torch.exp(all_likelihood - all_likelihood_max.unsqueeze(1))
                self_probs /= torch.sum(self_probs, dim=-1, keepdim=True)

                atoms = batched_data["x"][:, :, 0]
                mask = (~atoms.eq(0)).unsqueeze(1).unsqueeze(2).float()

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

                ref_probs = self.calc_kde_probs_from_rmsd(rmsds_ref)  # B x N1
                kl_div = torch.sum(
                    self_probs * torch.log(self_probs / (ref_probs + 1e-32)), axis=1
                ) / (torch.sum(self_probs, dim=1) + 1e-32)
                kl_div_sum = torch.sum(kl_div)
                return kl_div_sum
