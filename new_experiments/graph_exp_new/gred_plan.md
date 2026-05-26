## Plan: Unify GRED Config Across Pipelines
Minimal-change approach: keep each pipeline’s training logic intact, add backbone-aware plumbing and shared config/CLI compatibility so staged, three-staged, e2e, and indist all run with the same GRED-like config family.

**Steps**
1. Freeze a shared compatibility contract for common args: backbone, LapPE, GRED params, generator params, runtime params.
2. Patch parser surfaces across entrypoints to accept the shared contract with backward-compatible aliases. Depends on Step 1.
3. Make loaders backbone-aware everywhere using is_gred = backbone in (gred, hybrid), with dist masks only when needed. Depends on Step 2.
4. Replace remaining hardcoded GraphTransformer construction with existing build_model-style dispatch in missing scripts. Depends on Step 3.
5. Normalize forward calls so dist_masks/node_masks are passed only for gred/hybrid paths, preserving current stage behavior. Depends on Step 4.
6. Add checkpoint metadata guard: save backbone and validate on load to prevent cross-backbone resume mistakes. Parallel with Step 5.
7. Update run scripts to execute a controlled backbone matrix and isolate output directories by backbone. Depends on Steps 2-5.
8. Extend smoke/regression coverage in pipeline verification to include gred and hybrid paths. Depends on Steps 3-7.
9. Generate the plan document artifact in-repo as part of implementation handoff (this request’s deliverable).

**Relevant files**
- main_staged.py — main missing backbone-aware plumbing.
- main_indist.py — parser alias cleanup and consistent dist-mask gating.
- main_e2e.py — reuse as reference for backbone dispatch conventions.
- main_three_staged.py — align parser/loader behavior to shared contract.
- main_three_staged_multigpu.py — mirror three-staged compatibility in DDP path.
- verify_pipelines.py — add explicit gred/hybrid smoke tests.
- run_staged.sh — add backbone matrix and per-backbone dirs.
- run_three_staged.sh — same matrix strategy.
- run_e2e.sh — add gred no-proxy and hybrid proxy variants.
- run_indist.sh — align examples/defaults with shared contract.
- run_gred.sh — keep as baseline launcher, align naming and save layout.
- models.py — no major refactor in this task; usage-level compatibility only.

**Verification**
1. Parser compatibility test in each entrypoint with YAML + CLI override precedence.
2. One short smoke run per pipeline for vanilla_gt, gred (no-proxy), and hybrid (proxy-enabled).
3. Dist-mask shape and batching assertions in train/val/test loops.
4. Checkpoint load safety test: matching backbone passes, mismatched backbone fails fast with clear error.
5. Run-script dry/smoke matrix validation with backbone-separated output paths.
6. Vanilla regression check to confirm current commands still work unchanged.

**Decisions**
- Keep changes minimal and localized; no architecture rewrite in this task.
- Keep backward compatibility by adding aliases, not breaking current flags.
- Keep standalone gred as no-proxy path; use hybrid where proxy integration is required.
- Scope is interoperability in graph_exp_new pipelines only.