Approach: Self-Novelty Proxy Pipeline (New File)                                                                                                     
                                                                                                                                                                                                                                
Context                                                                                                                 
                                                                                                                                                                                                                                
Across all existing pipelines (main_indist.py, main_staged.py, main_three_staged.py), proxy generators converge to trivial proxies — embeddings too similar to the existing node distribution, contributing almost nothing new. The
failure modes differ per pipeline:

- main_indist.py Phase 3: Chamfer distance to dropped nodes fixes proxies to the node distribution.
- main_staged.py Stage 3: Regression to static Stage 2 targets removes any dynamic learning signal.
- main_three_staged.py Stage 2 / joint finetuning: Task loss alone finds the cheapest solution, which is proxies redundant with node information.

Decision: Build a new pipeline from scratch (alongside the existing three), centered on a self-novelty signal — same current model with vs without proxies — as a soft constraint alongside task loss. No teacher matching
anywhere. No modifications to existing pipelines.

---
Task-Specific Note: Multi-Label Classification

Peptides-func is multi-label: 10 independent binary classes, output is a multi-hot vector, loss is BCE on per-class logits. All novelty measurements must respect this:

- Use sigmoid per class, not softmax. p = sigmoid(logits / T) gives independent per-class probabilities in [0, 1].
- Divergences that assume a probability simplex (KL, JSD) don't apply cleanly. We use L2 on sigmoid outputs instead, which is correct for independent Bernoulli predictions.
- Temperature T still works as expected: higher T softens per-class probabilities, lower T sharpens them.

---
Pipeline

- File: main_self_novelty.py
- Stage 1: backbone pretrain
- Stage 2: frozen backbone, generator only, two forwards per batch (with / without proxies)
- Stage 3: joint finetune, same loss, no proxy dropout

---
Core Formulation
Normal training in Stage 1. Refere phase 1 of main_indist.py/main_three_staged.py.

In stage 2, for each training batch, run the same current model twice — once with proxies, once without — and measure how much the proxies changed its behavior. The no-proxy forward is a dynamic, self-referential baseline.

Output-level novelty (multi-label):
p_with    = sigmoid(logits_with / T)       # (B, C), each entry in [0, 1]
p_without = sigmoid(logits_without / T)       # (B, C), no grad
per_sample_l2 = || p_with - p_without ||_2 / sqrt(C)   # (B,), each in [0, 1]
output_novelty = mean over batch (per_sample_l2)        # scalar in [0, 1]
output_penalty = 1 - output_novelty

Node-level novelty (representation-level):
# Node embeddings AFTER all transformer layers and ALL proxy insertion points
# (i.e., final representation of the original N nodes — exclude proxy tokens).

node_novelty = mean over original nodes of
            (1 - cosine_similarity(node_emb_with, node_emb_without))
            # bounded in [0, 2]

node_penalty = 2 - node_novelty

Combined generator loss:
L_gen = task_loss(BCEWithLogits)  +  alpha * output_penalty  +  alpha_node * node_penalty

Defaults (all configurable):
- Temperature T = 1.0
- alpha = 1.0
- alpha_node = 1.0
- No alpha scheduling — constant throughout training

Properties:
- Bounded below by 0. Task loss and both penalties are non-negative.
- Self-regulating. Once novelty ≥ bound, the penalty is 0 and its gradient is 0; optimizer redirects effort to task loss.
- Symmetric. L2 distance and cosine distance are both direction-independent.
- Multi-label-correct. Sigmoid + L2 for independent binary classes.
- Scale-matched. Task loss ~O(0.1–2.0), penalties ~O(0-2), alpha=1.0 keeps them comparable.

---
Where the Node-Level Comparison Is Computed

For models with multiple proxy insertion points (MultiPointProxyWrapper with proxy_insertion_layers = [0, 2, 4] etc.):

- The "with proxies" forward runs the full wrapper — proxies are injected at every configured insertion point.
- The "without proxies" forward runs the same model with the wrapper bypassed entirely — no proxies at any layer.
- The node representations compared for node_novelty are the final representations of the original N nodes, after the last transformer layer — i.e., the representations used for readout pooling.

This captures the cumulative effect of all proxy insertions on node representations.

---
New Pipeline Structure

File: main_self_novelty.py (placed alongside existing main_*.py files in new_experiments/graph_exp_new/).

Two-stage design

Stage 1: Backbone pretraining

- Train transformer / GRED / hybrid backbone on full N-node graphs.
- Standard BCE-with-logits task loss.
- Identical to existing Stage 1 / Phase 1 elsewhere — can reuse checkpoints from existing pipelines if architecture matches.
- Output: backbone checkpoint.

Stage 2: Generator-only training (frozen backbone)

- Backbone frozen (all parameters, eval mode, no grad).
- Generator trainable.
- Each batch:
a. Encode nodes through frozen backbone → (dense_x, mask) (cached / shared across both forwards).
b. Generate proxies from node embeddings.
c. Two forwards through the frozen backbone:
   - With proxies: proxies injected per config (single-point or multi-point). Returns logits_with, node_emb_with.
 - Without proxies: wrapper bypassed entirely. Returns logits_without, node_emb_without (both detached — no grad needed).
d. Compute L_gen = task_loss + alpha * output_penalty + alpha_node * node_penalty.
e. Backprop to generator only.
- Output: generator checkpoint.

Stage 3: Joint finetune (both trainable)

- Backbone unfrozen.
- Both backbone and generator trainable, differential learning rates (initially same, modifiable by inputs).
- Same per-batch structure as Stage 2, except:
- The "without proxies" forward is still done in torch.no_grad() — its gradient is not needed; only the "with proxies" forward contributes to gradients.
- Loss formulation unchanged: task_loss + alpha * output_penalty + alpha_node * node_penalty.
- No proxy dropout. (Confirmed this isn't wanted in the new pipeline.)
- Output: final finetuned checkpoint.

What's explicitly not in this pipeline

- No Phase 2 / N-M subsampling step (main_indist.py-style partial-graph training).
- No Stage 2 per-graph proxy optimization (main_staged.py-style static targets).
- No Chamfer / MMD / reconstruction losses of any kind.
- No teacher checkpoint beyond Stage 1's weights used for initialization.
- No proxy_dropout mechanism.

---
Implementation Notes

- Two forwards per batch: the "without proxies" forward is extra compute. Share the node-encoding step between the two (run encoder once, reuse dense embeddings) — this is already supported via precomputed_dense= on existing
models, so no new infra required.
- Bypassing multi_point_proxy: existing models auto-inject proxies when multi_point_proxy is attached. For the "without proxies" forward, the cleanest approach is a boolean flag on the forward method (e.g.
disable_proxy_injection=True) that skips the wrapper's insertion logic. About 5 lines of change per model class. Refer existing pipelines to identify how without proxy inputs were passed in their phase 1 or non proxy phases.
- Per-graph node novelty: compute cosine similarity per node, average only over the B × N original-node positions (exclude padding; proxies never appear in the original-node tensor).
- Backbone choices: all three existing backbones (vanilla_gt, gred, hybrid) should be supported. The novelty formulation is backbone-agnostic.
- Generator choices: any existing generator (score_based, pma, graph_coarsening, gnn_pooling, flow_matching (trained via only task loss+penalties)) — no new generator types needed. The formulation is generator-agnostic.
- Backbone flag for bypass. Should the disable_proxy_injection bypass be added to GraphTransformer, GREDEncoder, and GREDHybridTransformer in models.py  
The multi_point_proxy wrapper is only attached to the model when that phase actually needs proxies. In Phase 1 and Phase 2 of main_indist.py, for example, the model is instantiated with multi_point_proxy=None (no wrapper attached at construction), so the auto-inject condition at models.py is false and the forward takes the clean "no proxies" branch. The bypass flag only becomes necessary in our new pipeline when a single model instance (with multi_point_proxy already attached at construction, for multi-point configs) needs to run both a with-proxies and without-proxies forward in the same batch.

---
Extra Logging per epoch:

- Log inter-proxy mean+std of cosine similarity. This helps identify if generated proxies are different enough or if some penalty to make them diverse is needed.
- Log node and output penalties separately that are added to the task loss.

---
What This Approach Does NOT Do

- No teacher matching (removed from all phases of the new pipeline).
- No MMD-to-nodes constraint — that would pull proxies toward the node distribution, the opposite of what we want.
- No reconstruction against dropped nodes.
- No proxy dropout.
- No modifications to main_indist.py, main_staged.py, main_three_staged.py, or main_e2e.py.

