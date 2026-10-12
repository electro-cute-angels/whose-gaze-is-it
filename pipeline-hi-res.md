# Hi-Res BPJDet Pipeline Notes

This document explains your current GazeTarget360 workflow for the hi-res dataset, how it differs from the original upstream pipeline, what the heatmaps represent, how overlays are produced, and how threshold tuning evolved through testing.

## 1) How your hi-res pipeline works

Main baseline hi-res config:

- `configs/pipeline.bpjdet.hires.yaml`

Key fields:

- `input_root`: hi-res image folder
- `output_root`: dedicated output folder
- `checkpoint_path`: resumable progress store
- `detector.name: bpjdet_head`: custom head detector front-end

Execution path:

1. `scripts/run_archive_parallel.py` loads config and shards image paths across workers/GPUs.
2. Each worker loads images from `input_root`, runs head detection and gaze inference, and writes shard outputs.
3. The master process merges shard outputs into:
   - `predictions.jsonl`
   - `predictions.parquet`
   - `runtime_metrics.json`
4. Checkpoints prevent reprocessing of already completed images.

Core files:

- `scripts/run_archive_parallel.py`
- `src/gazetarget360_pipeline/inference/gt360_adapter.py`
- `src/gazetarget360_pipeline/detectors/bpjdet_head.py`
- `src/gazetarget360_pipeline/output_writer.py`

## 2) How this differs from original GazeTarget360

Upstream flow (`third_party/DisengageNet`) usually runs conditional EC + GT360 inference with dlib/DETR-style face detection and demo-oriented scripts.

Your pipeline differs in the following ways:

1. Detector replacement:
- Original: dlib/DETR-style front-end in upstream demos.
- Yours: BPJDet crowd-head detector (`bpjdet_head`) via SemiUHPE ecosystem.

2. Crowded-scene localization:
- BPJDet is trained for crowd heads (class id 1), yielding tighter head boxes than generic person-to-head heuristics.

3. Archive-scale architecture:
- Resumable, sharded, multi-worker pipeline for large corpora.

4. Native-resolution processing:
- Images are loaded at original resolution; detector inference is letterboxed then projected back to original coordinates.

5. Structured output schema:
- Per-head rows include confidence, class probabilities, gaze point, heatmap path, detector name, and run id.

6. Extended thresholding logic (new):
- Original and early custom versions used a single EC threshold and single in/out threshold.
- Final tuned version supports an EC gray-zone (dual EC thresholds plus dual in/out thresholds) for better behavior on borderline cases.

## 3) Gaze decision logic (current and historical)

Per detected head, `GT360Adapter` computes:

- `ec_prob`
- `inout_prob`

And probabilities:

- `p_EC = ec_prob`
- `p_IFT = (1 - ec_prob) * inout_prob`
- `p_OFT = (1 - ec_prob) * (1 - inout_prob)`

### Historical rule (baseline and first tuning passes)

1. if `p_EC >= ec_threshold` => `EC`
2. else if `inout_prob >= inout_threshold` => `IFT`
3. else => `OFT`

### Final rule (gray-zone EC logic)

1. if `p_EC >= ec_threshold` => `EC`
2. else if `p_EC >= ec_threshold_low` => use stricter `inout_threshold`
3. else (`p_EC < ec_threshold_low`) => use relaxed `inout_threshold_low`

This final rule is implemented in:

- `src/gazetarget360_pipeline/inference/gt360_adapter.py`

And wired in:

- `scripts/run_archive_parallel.py`
- `src/gazetarget360_pipeline/pipeline.py`

## 4) Experiment log: threshold tuning journey

All tests below used 10-image pilots plus overlay inspection.

### A) Baseline hi-res run

- Config: `configs/pipeline.bpjdet.hires.yaml`
- Thresholds: `ec_threshold=0.5`, `inout_threshold=0.5`
- Output: `outputs_hires_bpjdet`

Observed issue:

- Some borderline heads were classified as `EC` when visual inspection suggested `IFT` or `OFT`.

User-provided examples:

- `V-P-NG-D-00001-07_gaze.jpg`: expected IFT for main head, baseline predicted EC.
- `V-P-NG-D-00001-10_gaze.jpg`: expected OFT for main head, baseline predicted EC.

### B) Single-threshold tuning (tuned)

- Config: `configs/pipeline.bpjdet.hires.tuned.yaml`
- Thresholds: `ec_threshold=0.72`, `inout_threshold=0.60`
- Output: `outputs_hires_bpjdet_tuned`

Why chosen:

- Sweep showed this pair fixed both cited examples (07 main head became IFT; 10 main head became OFT).

Regression found:

- `V-P-NG-D-00001-03_gaze.jpg` became worse versus baseline.
- In particular, one borderline head with `p_EC=0.717...` flipped away from EC at 0.72.

### C) Single-threshold refinement (tuned2)

- Config: `configs/pipeline.bpjdet.hires.tuned2.yaml`
- Thresholds: `ec_threshold=0.705`, `inout_threshold=0.60`
- Output: `outputs_hires_bpjdet_tuned2`

Effect:

- Restored EC on the borderline head in `00001-03`.
- Kept the two major fixes (`00001-07` and `00001-10`) on their main heads.
- Still showed tradeoffs in neighboring borderline heads because one global in/out threshold was still too rigid.

### D) Final gray-zone tuning (tuned3, current best)

- Config: `configs/pipeline.bpjdet.hires.tuned3.yaml`
- Thresholds:
  - `ec_threshold=0.71`
  - `ec_threshold_low=0.60`
  - `inout_threshold=0.60`
  - `inout_threshold_low=0.50`
- Output: `outputs_hires_bpjdet_tuned3`

Rationale:

- One EC threshold could not satisfy all three images simultaneously.
- The gray-zone strategy treats mid-EC-probability heads separately from clearly non-EC heads.

Result on key test cases:

- `00001-07` main head: EC -> IFT (desired)
- `00001-10` main head: EC -> OFT (desired)
- `00001-03` problematic head: preserved as EC (closer to baseline behavior judged more accurate)

Conclusion:

- `tuned3` is the best compromise discovered in pilot testing and is currently the recommended config for full hi-res processing.

## 5) What the heatmaps are

For each detected head, GT360 outputs a 2D heatmap indicating likely gaze target regions.

- Typical shape for this model family is around 64x64.
- Higher values indicate higher gaze-target likelihood.

Stored as compressed NumPy arrays:

- `output_root/heatmaps/<image_id>_head<head_id>.npy.gz`

Writer:

- `save_heatmap(...)` in `src/gazetarget360_pipeline/output_writer.py`

Behavior:

- `IFT`: gaze point produced from heatmap
- `EC` and `OFT`: no gaze point (null by design)

## 6) How gaze points are extracted from heatmaps

In `GT360Adapter`:

1. Peak cell via `argmax`
2. Soft-argmax refinement (weighted centroid in peak neighborhood)
3. Convert to normalized coordinates `[0, 1] x [0, 1]`

This reduces grid quantization effects and gives smoother target points.

## 7) How to use heatmaps in practice

1. Visual QA:
- Confirm whether predicted target region matches scene semantics.

2. Confidence diagnostics:
- Diffuse maps suggest ambiguity.
- Sharp peaks suggest strong target localization.

3. Filtering:
- Use `heatmap_peak_value` and class probabilities for stricter acceptance criteria.

4. Cross-run comparison:
- Compare baseline/tuned/tuned2/tuned3 on the same filenames for controlled analysis.

## 8) How overlays are created

Overlay script:

- `scripts/visualize_results.py`

Input:

- `--predictions` (`predictions.jsonl` or `predictions.parquet`)

Outputs:

1. Full-frame annotated boards:
- `<image>_gaze.jpg`

2. Per-head panels:
- `heads/<image>_headN.jpg`

Rendering style:

- IFT = green, OFT = red, EC = blue
- IFT heads get arrow and target marker
- Heatmaps are upscaled and blended into visualization panels

## 9) Output folders used in this tuning cycle

1. Baseline:
- `outputs_hires_bpjdet`

2. Single-threshold tuned:
- `outputs_hires_bpjdet_tuned`

3. Single-threshold refined:
- `outputs_hires_bpjdet_tuned2`

4. Final gray-zone tuned (recommended):
- `outputs_hires_bpjdet_tuned3`

Each output root contains:

- `predictions.jsonl`
- `predictions.parquet`
- `runtime_metrics.json`
- `heatmaps/`
- `overlays/`
- `checkpoints/processed_images.jsonl`

## 10) Recommended run sequence from this point

1. Keep `configs/pipeline.bpjdet.hires.tuned3.yaml` as the default for production hi-res runs.
2. Launch full run without `--max-images`.
3. Keep the same output root for resume safety.
4. Spot-check overlays periodically (especially borderline crowd scenes) before final export.

## 11) Freeze policy and validation protocol

This section defines how we run experiments from now on so decisions remain traceable.

### Freeze policy

When a logic variant appears best, we freeze it in a dedicated config and do not mutate it during validation.

Frozen tuned3 config:

- `configs/pipeline.bpjdet.hires.tuned3.frozen.yaml`

Meaning:

- Threshold logic is locked for evaluation.
- Any further ideas are tested in new configs/output roots, never by editing the frozen one.

### Why we validate on random subsets

The 10-image pilot is useful to iterate quickly, but it can overfit to a few scenes.

A larger random validation set helps answer:

1. Does tuned behavior generalize beyond known edge cases?
2. Are we trading one type of error for another at scale?
3. Is the method stable enough for full-dataset execution?

### Current validation batch (in progress)

Dataset sample:

- Subset folder: `data/validation/hires_rand60_seed42`
- Manifest: `data/validation/hires_rand60_seed42_manifest.txt`
- Sampling method: random sample, seed 42, from hi-res corpus
- Size: 60 images

Validation config:

- `configs/pipeline.bpjdet.hires.tuned3.val60.seed42.yaml`

Validation outputs:

- `outputs_hires_bpjdet_tuned3_val60_seed42`

Goal of this batch:

1. Manual human QA on a broader random set.
2. Confirm that tuned3 remains accurate outside the initial troubleshooting images.
3. Decide go/no-go for full hi-res run using frozen tuned3 logic.

### Decision rule after manual review

After manual review of this 60-image batch:

1. If quality is consistently good, proceed to full run with frozen tuned3 logic.
2. If new systematic errors appear, branch to a new config (tuned4, etc.) and repeat the same protocol.

## 12) Experiment journal (living log)

This is the running narrative of why each experiment was executed and what we expected to learn.

### Journal entry: baseline hi-res pilot

Why:

- Establish a first hi-res reference output with BPJDet integration.

Hope:

- Verify technical compatibility and spot initial EC/IFT/OFT behavior.

Outcome:

- Pipeline worked end-to-end, but borderline EC errors were observed.

### Journal entry: tuned (single-threshold aggressive)

Why:

- Fix clear false-EC examples by raising EC threshold and tightening in/out split.

Hope:

- Correct known errors in `00001-07` and `00001-10`.

Outcome:

- Targeted fixes succeeded, but regression appeared on `00001-03`.

### Journal entry: tuned2 (single-threshold compromise)

Why:

- Reduce regression by lowering EC threshold slightly.

Hope:

- Keep the two fixes while restoring better behavior in `00001-03`.

Outcome:

- Improved compromise, but still constrained by one global threshold regime.

### Journal entry: tuned3 (gray-zone logic)

Why:

- Evidence showed one EC threshold could not satisfy all borderline scenarios.

Hope:

- Treat mid-EC and low-EC cases differently to preserve good EC calls while fixing false EC.

Outcome:

- Best observed behavior on key images; selected as current candidate.

### Journal entry: frozen tuned3, random-60 validation (current)

Why:

- Move from anecdotal examples to broader, reproducible manual evaluation.

Hope:

- Confirm tuned3 generalization and justify full-run launch.

Status:

- In manual review phase.

Generated review artifacts (head-level):

- `outputs_hires_bpjdet_tuned3_val60_seed42/review/head_level_review.csv`
- `outputs_hires_bpjdet_tuned3_val60_seed42/review/high_priority_non_ec.csv`
- `outputs_hires_bpjdet_tuned3_val60_seed42/review/review_summary.json`

Current review package stats:

- total rows: 420
- total images: 60
- predicted class counts: IFT 247, OFT 131, EC 33, empty 9 (rows without valid head label)
- priority counts: high 53, medium 45, low 322

How manual labeling should be done:

1. Start with `high_priority_non_ec.csv` (most likely disagreements).
2. Fill `human_label` with `EC`, `IFT`, or `OFT`.
3. Fill `human_confidence` (suggested 1-5 scale).
4. Add rationale in `review_notes` for ambiguous cases.
5. Continue with medium priority rows if needed.

### Journal entry: random-60 manual QA feedback (new)

Reviewer feedback (examples):

- `V-P-NG-E-00365_gaze.jpg`: reviewer believes multiple heads are eye contact.
- `V-P-NG-E-00422_gaze.jpg`: reviewer believes eye contact.
- `V-P-NG-N-00003-69_gaze.jpg`: at least one OFT likely eye contact.
- `V-P-NG-N-00012-28A_gaze.jpg`: at least one IFT likely eye contact.
- `V-P-NG-N-00039-13_gaze.jpg`: several OFT likely eye contact.
- `V-P-NG-N-00044-24_gaze.jpg`: at least one OFT likely eye contact.

Why this matters:

- This is the first broader random-batch signal that tuned3 still under-calls EC in some scenes.

What probability analysis showed:

1. Some false non-EC predictions are borderline and potentially threshold-fixable:
- Example: OFT rows with moderate/high `p_EC` around ~0.44 to ~0.65.

2. Several reviewer-disputed cases have low `p_EC` (very far from EC threshold):
- In these cases, changing thresholds alone is unlikely to recover EC without creating new regressions elsewhere.

Interpretation:

- We have two error regimes:
  - Regime A (borderline): threshold calibration can help.
  - Regime B (low-EC-score but visually EC): EC model/domain mismatch, not just thresholding.

Decision implication:

- Do not rely only on more threshold sweeps.
- Keep tuned3 frozen for reproducibility while building a labeled review set to quantify where EC model fails.

Planned next step after this review batch:

1. Build head-level manual labels on a subset of disputed examples.
2. Compute confusion and calibration diagnostics for EC vs non-EC.
3. Decide between:
   - minor threshold refinement for Regime A,
   - EC model adaptation/fine-tuning strategy for Regime B.

How this document will be maintained:

- Every new experiment gets a journal entry (why, hope, outcome, decision).
- We do not overwrite history; we append to preserve reasoning lineage.

### Journal entry: review-webapp for head-level annotation (new)

Why:

- Manual CSV editing is too slow and error-prone for large head-level review batches.

Hope:

- Make annotation faster and more consistent by providing a side-by-side visual UI with direct box selection and immediate saves.

Implementation:

- New app folder: `review-webapp/`
- Backend: `review-webapp/app.py` (Flask)
- Frontend: `review-webapp/static/index.html`, `review-webapp/static/style.css`, `review-webapp/static/app.js`

Behavior:

1. Loads review index from `head_level_review.csv`.
2. Shows original image (left) and overlay (right).
3. Draws clickable head bounding boxes in both panels.
4. Lets reviewer set `human_label`, `human_confidence`, and `review_notes` per head.
5. Persists edits to `annotations.webapp.json`.
6. Writes merged `head_level_review_annotated.csv` continuously.
7. Supports CSV export from UI endpoint.

Design notes:

- Visual language intentionally mirrors the existing `headpose-webapp` style:
  grayscale UI, yellow accent striping, uppercase section headers, editorial archive feel.

Status:

- Backend API smoke-tested with Flask test client (`/api/index` and `/api/image` successful).
- Ready for interactive manual review in browser.

Decision:

- Use the webapp as the primary labeling tool for the random-60 validation set and subsequent larger validation subsets.

### Journal entry: annotation-driven threshold sweep (new)

Why:

- We now have reviewer labels on the random-60 batch and can tune thresholds against real human decisions instead of only visual spot checks.

Data used:

- Reviewed file: `head_level_review_annotated.csv`
- Usable labeled heads (EC/IFT/OFT/unsure): 411
- Strict labeled heads for threshold scoring (EC/IFT/OFT only): 398
- Label mix: IFT 245, OFT 84, EC 69, unsure 13

Current tuned3 score on reviewed labels:

- Thresholds: `ec_threshold=0.71`, `ec_threshold_low=0.60`, `inout_threshold=0.60`, `inout_threshold_low=0.50`
- Macro F1 (EC/IFT/OFT): 0.6518
- Per-class F1: EC 0.4752, IFT 0.8589, OFT 0.6214

Sweep findings:

1. Best pure-F1 point from broad grid search:
- `ec_threshold=0.61`, `ec_threshold_low=0.45`, `inout_threshold=0.60`, `inout_threshold_low=0.51`
- Macro F1: 0.6651

2. Constrained best point (keeps EC threshold closer to tuned3 while still improving):
- `ec_threshold=0.70`, `ec_threshold_low=0.45`, `inout_threshold=0.59`, `inout_threshold_low=0.51`
- Macro F1: 0.6634

Interpretation:

- Reviewer labels support lowering the effective EC barrier to reduce under-called EC.
- Improvement is moderate but consistent versus tuned3 on this validation batch.
- Detector-confidence pruning alone is not a clean fix: raising detector threshold removes many true IFT heads.

New validation configs created:

1. Conservative tuned4 candidate:
- `configs/pipeline.bpjdet.hires.tuned4.val60.seed42.yaml`
- Uses constrained-best thresholds (`0.70 / 0.45 / 0.59 / 0.51`)

2. EC-recall oriented tuned4 candidate:
- `configs/pipeline.bpjdet.hires.tuned4.ec_recall.val60.seed42.yaml`
- Uses stronger EC recovery (`0.66 / 0.52 / 0.60 / 0.51`)

Decision for next step:

- Run both tuned4 variants on the same random-60 subset.
- Compare disagreement rows and reviewer acceptance, especially for previous under-called EC scenes.
- If both remain imperfect on low-EC-score visual EC cases, prioritize EC model adaptation over further threshold-only sweeps.

### Journal entry: tuned4 validation launch attempt and blocker (new)

Why:

- We moved from threshold-selection analysis to execution: run both tuned4 variants on the same 60-image subset and compare measured agreement with human labels.

Attempted runs:

1. Conservative tuned4:
- Config: `configs/pipeline.bpjdet.hires.tuned4.val60.seed42.yaml`
- Output root: `outputs_hires_bpjdet_tuned4_val60_seed42`

2. EC-recall tuned4:
- Config: `configs/pipeline.bpjdet.hires.tuned4.ec_recall.val60.seed42.yaml`
- Output root: `outputs_hires_bpjdet_tuned4_ec_recall_val60_seed42`

Observed blocker:

- Both runs failed before generating predictions due to a PyTorch 2.6 checkpoint loading change in SemiUHPE dependency code.
- Error family:
  - `_pickle.UnpicklingError: Weights only load failed`
  - `Unsupported global: numpy.core.multiarray._reconstruct`
- Failure point seen in stack trace:
  - `SemiUHPE/models/experimental.py`, call to `torch.load(...)` in `attempt_load`

Impact:

- No `predictions.jsonl` produced for either tuned4 output root.
- Immediate tuned3 vs tuned4 accuracy comparison on fresh predictions could not be executed yet.

What this tells us:

- The threshold logic/configs are ready, but current runtime environment + upstream loader compatibility must be fixed before validation can continue.

Next validation steps (execution-side):

1. Apply a safe compatibility fix for PyTorch 2.6 loading in the SemiUHPE loader path used by BPJDet.
2. Re-run both tuned4 validation configs.
3. Produce side-by-side tuned3/tuned4 confusion tables on the reviewed labeled subset.
4. Record final go/no-go decision for tuned4 promotion.

### Journal entry: tuned4 rerun and measured comparison (new)

Why:

- After the launch blocker, we re-ran both tuned4 validation variants to obtain measurable agreement with reviewed labels.

Execution notes:

1. Initial rerun attempt with system `python3` failed due to missing dependency:
- `ModuleNotFoundError: pyarrow`

2. Rerun with project environment succeeded:
- Interpreter: `.venv/bin/python`
- Both tuned4 pipelines completed and wrote predictions.

Compared runs:

1. `outputs_hires_bpjdet_tuned4_val60_seed42`
2. `outputs_hires_bpjdet_tuned4_ec_recall_val60_seed42`

Evaluation set:

- Human labels from reviewed CSV (`EC/IFT/OFT` only)
- Matched labeled heads: 398

Results:

1. Conservative tuned4 (`ec=0.70`, `ec_low=0.45`, `inout=0.59`, `inout_low=0.51`)
- Accuracy: 0.7563
- Macro F1: 0.6634

2. EC-recall tuned4 (`ec=0.66`, `ec_low=0.52`, `inout=0.60`, `inout_low=0.51`)
- Accuracy: 0.7538
- Macro F1: 0.6606

Decision from this comparison:

- Conservative tuned4 is currently the better overall variant on the reviewed random-60 set.
- Keep EC-recall variant as alternate for targeted qualitative checks, but promote conservative tuned4 as default next candidate.

Next step:

1. Run overlay-level qualitative review focused on known EC under-call scenes.
2. If qualitatively consistent, freeze conservative tuned4 and prepare full-run config.

### Journal entry: tuned3 vs conservative tuned4 delta (new)

Why:

- We needed an explicit, apples-to-apples answer for how much conservative tuned4 differs from tuned3 on the same reviewed labels.

Evaluation protocol:

- Same random-60 reviewed annotation file.
- Same labeled subset: 398 heads (EC/IFT/OFT only).
- Compared predictions from:
  - `outputs_hires_bpjdet_tuned3_val60_seed42/predictions.jsonl`
  - `outputs_hires_bpjdet_tuned4_val60_seed42/predictions.jsonl`

Metric deltas (tuned4 conservative minus tuned3):

- Accuracy: `0.7563 - 0.7487 = +0.0075`
- Macro F1: `0.6634 - 0.6518 = +0.0116`

Prediction-level change count:

- Total heads with changed class prediction: 5 / 398
- Outcome on changed heads:
  - Improved: 3
  - Worsened: 0
  - Sideways (changed class but still wrong): 2

Interpretation:

- Conservative tuned4 is a small, targeted improvement over tuned3 rather than a broad behavior shift.
- Gains come from a handful of borderline heads with no observed regressions on changed cases in this reviewed subset.

### Journal entry: recommended forward path after tuned4 comparison (new)

Why:

- Conservative tuned4 shows measurable gains, but the delta is small and we need stronger evidence before full-corpus promotion.

Recommended sequence:

1. Freeze conservative tuned4 in a dedicated config and do not mutate it during validation.
2. Run a second random validation subset (larger than 60 images, e.g., 200-300) with a new seed.
3. Evaluate tuned3 vs tuned4 only on human-labeled heads and collect class-wise deltas (especially EC recall).
4. Build a disagreement review pack (rows where tuned3 and tuned4 differ) for fast manual adjudication.
5. If tuned4 gains persist on the larger subset, promote tuned4 as default for full hi-res run.
6. In parallel, start an EC-focused adaptation track for low-EC-score but visually-EC cases, since thresholding alone cannot fully solve Regime B errors.

Decision gate:

- Promote tuned4 only if it improves macro F1 and EC recall on the second validation subset without meaningful regression in OFT/IFT precision.

### Journal entry: heads-only random-200 evaluation package (new)

Why:

- We needed a larger validation batch for collaborative annotation, with a hard constraint that images with no detected heads are excluded from the evaluation set.

Execution protocol:

1. Created a seed-fixed pre-sample pool to guarantee enough head-positive images:
- Subset folder: `data/validation/hires_rand260_seed20260915_pool`
- Manifest: `data/validation/hires_rand260_seed20260915_manifest.txt`
- Config: `configs/pipeline.bpjdet.hires.tuned4.val260.seed20260915.yaml`

2. Ran conservative tuned4 on the 260-image pool:
- Output root: `outputs_hires_bpjdet_tuned4_val260_seed20260915`
- Runtime: ~71.8s
- Predictions rows: 2502

3. Derived the final evaluation set from head-positive images only:
- Final manifest: `data/validation/hires_rand200_heads_only_seed20260915_manifest.txt`
- Final subset folder: `data/validation/hires_rand200_heads_only_seed20260915`
- Final size: 200 images
- Inclusion rule: at least one detected head (`head_id != null`)
- Exclusion rule: any no-head image is not included in this 200-image set.

4. Added reproducible config for reruns on the finalized 200 set:
- `configs/pipeline.bpjdet.hires.tuned4.val200_heads.seed20260915.yaml`

5. Built a self-contained shareable annotation package:
- Package root: `share/review_eval200_heads_tuned4_seed20260915`
- Contains:
  - `review-webapp/` (app + static assets + launch script)
  - `data/eval200/images/` (the 200 selected source images)
  - `run/predictions.jsonl` (selected rows only, no null-head rows)
  - `run/overlays/` and `run/overlays/heads/`
  - `run/review/head_level_review.csv`
  - `run/review/high_priority_non_ec.csv`
  - `run/review/review_summary.json`

Package stats:

- Review rows (heads to annotate): 2197
- High-priority non-EC rows: 837

Operational note:

- Package `review-webapp/run.sh` now launches the app with `GAZE_REVIEW_RUN_ROOT=../run`, so collaborators can annotate directly inside the shared folder without local path edits.

### Journal entry: quantified BPJDet detector effectiveness (new)

Why:

- We needed a concrete, numeric summary of how much BPJDet improved head detection versus prior detector variants.

Method:

1. Read `predictions.jsonl` from completed runs.
2. Count only rows with `head_id != null` as detected heads.
3. For fair detector comparison, use overlapping image basenames between runs.

Direct matched-set detector comparisons:

1. `outputs_bpjdet` vs `outputs_dlib` (same 10-image overlap)
- BPJDet heads: 98
- Dlib heads: 24
- Absolute gain: +74 heads
- Relative gain: +308.33%
- Mean heads/image: 9.8 (BPJDet) vs 2.4 (Dlib)

2. `outputs_bpjdet` vs `outputs_dlib_up2` (same 10-image overlap)
- BPJDet heads: 98
- Dlib-up2 heads: 37
- Absolute gain: +61 heads
- Relative gain: +164.86%
- Mean heads/image: 9.8 (BPJDet) vs 3.7 (Dlib-up2)

Interpretation:

- On the directly comparable 10-image set, BPJDet detects substantially more heads than both Dlib baselines.
- The largest gain is versus plain Dlib (+74 heads, about 4.08x as many detections).
- BPJDet still shows a large gain versus the stronger Dlib-up2 variant (+61 heads, about 2.65x as many detections).

Additional BPJDet coverage indicators (non-baseline-matched runs):

1. Archive-scale BPJDet run (`outputs_archive_bpjdet`)
- Images: 562
- Detected-head rows: 5018
- Images with at least one detected head: 512
- Images with no detected heads: 50
- Mean heads/image: 8.93
- Mean heads per head-positive image: 9.80

2. Hi-res BPJDet 260-image validation pool (`outputs_hires_bpjdet_tuned4_val260_seed20260915`)
- Images: 260
- Detected-head rows: 2472
- Images with at least one detected head: 230
- Images with no detected heads: 30
- Mean heads/image: 9.51
- Mean heads per head-positive image: 10.75

Scope note:

- The strongest apples-to-apples detector effectiveness claims are the two matched 10-image comparisons above.
- Larger-run BPJDet-only statistics are included as supporting scale/coverage context, not as strict baseline-vs-baseline deltas.

### Journal entry: full accuracy metrics across tuned variants (new)

Why:

- We wanted a single place with comprehensive classification metrics for all tuned versions evaluated so far.

Evaluation protocol:

1. Human labels source:
- `outputs_hires_bpjdet_tuned3_val60_seed42/review/head_level_review_annotated.csv`

2. Label filtering:
- Keep rows where `human_label` is one of `EC`, `IFT`, `OFT` and `head_id` is numeric.

3. Join key:
- `basename(image_name or image_path) + head_id`

4. Metrics reported per model:
- `n_matched`
- accuracy
- balanced accuracy
- macro precision / recall / F1
- weighted precision / recall / F1
- micro precision / recall / F1
- per-class precision / recall / F1 (`EC`, `IFT`, `OFT`)
- multiclass MCC
- Cohen kappa
- confusion matrix (rows=true, cols=pred in `EC/IFT/OFT` order)

Important comparability note:

- `tuned3_val60`, `tuned4_val60`, and `tuned4_ec_recall_val60` each evaluate on 398 labeled heads.
- Earlier pilot outputs (`tuned`, `tuned2`, `hires_bpjdet`) overlap only 12 labeled heads, so they are useful as historical signals but not directly comparable in confidence.

Aggregate metrics:

| model | n_matched | accuracy | balanced_acc | macro_p | macro_r | macro_f1 | weighted_p | weighted_r | weighted_f1 | mcc | kappa |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tuned | 12 | 0.8333 | 0.6061 | 0.4444 | 0.6061 | 0.4667 | 0.9444 | 0.8333 | 0.8667 | 0.5222 | 0.4286 |
| tuned2 | 12 | 0.8333 | 0.6061 | 0.4444 | 0.6061 | 0.4667 | 0.9444 | 0.8333 | 0.8667 | 0.5222 | 0.4286 |
| tuned3_val60_seed42 | 398 | 0.7487 | 0.6556 | 0.7117 | 0.6556 | 0.6518 | 0.7705 | 0.7487 | 0.7423 | 0.5472 | 0.5381 |
| tuned4_val60_seed42 | 398 | 0.7563 | 0.6684 | 0.7206 | 0.6684 | 0.6634 | 0.7788 | 0.7563 | 0.7505 | 0.5630 | 0.5537 |
| tuned4_ec_recall_val60_seed42 | 398 | 0.7538 | 0.6670 | 0.7128 | 0.6670 | 0.6606 | 0.7759 | 0.7538 | 0.7486 | 0.5601 | 0.5509 |
| hires_bpjdet | 12 | 1.0000 | 0.6667 | 0.6667 | 0.6667 | 0.6667 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

Micro metrics note:

- For all runs above, micro precision = micro recall = micro F1 = accuracy (multiclass single-label setting).

Per-class precision / recall / F1 (`EC`, `IFT`, `OFT`):

| model | EC (P/R/F1) | IFT (P/R/F1) | OFT (P/R/F1) |
| --- | --- | --- | --- |
| tuned | 0.0000 / 0.0000 / 0.0000 | 1.0000 / 0.8182 / 0.9000 | 0.3333 / 1.0000 / 0.5000 |
| tuned2 | 0.0000 / 0.0000 / 0.0000 | 1.0000 / 0.8182 / 0.9000 | 0.3333 / 1.0000 / 0.5000 |
| tuned3_val60_seed42 | 0.7500 / 0.3478 / 0.4752 | 0.8607 / 0.8571 / 0.8589 | 0.5246 / 0.7619 / 0.6214 |
| tuned4_val60_seed42 | 0.7576 / 0.3623 / 0.4902 | 0.8678 / 0.8571 / 0.8624 | 0.5366 / 0.7857 / 0.6377 |
| tuned4_ec_recall_val60_seed42 | 0.7353 / 0.3623 / 0.4854 | 0.8708 / 0.8531 / 0.8619 | 0.5323 / 0.7857 / 0.6346 |
| hires_bpjdet | 0.0000 / 0.0000 / 0.0000 | 1.0000 / 1.0000 / 1.0000 | 1.0000 / 1.0000 / 1.0000 |

Confusion matrices (rows=true, cols=pred; class order = `EC/IFT/OFT`):

- tuned: `[[0,0,0],[0,9,2],[0,0,1]]`
- tuned2: `[[0,0,0],[0,9,2],[0,0,1]]`
- tuned3_val60_seed42: `[[24,15,30],[7,210,28],[1,19,64]]`
- tuned4_val60_seed42: `[[25,15,29],[7,210,28],[1,17,66]]`
- tuned4_ec_recall_val60_seed42: `[[25,14,30],[8,209,28],[1,17,66]]`
- hires_bpjdet: `[[0,0,0],[0,11,0],[0,0,1]]`

Delta summary vs tuned3 (`accuracy`, `macro_f1`, `EC recall`, `OFT precision`):

| model | d_accuracy | d_macro_f1 | d_EC_recall | d_OFT_precision |
| --- | ---: | ---: | ---: | ---: |
| tuned | +0.0846 | -0.1852 | -0.3478 | -0.1913 |
| tuned2 | +0.0846 | -0.1852 | -0.3478 | -0.1913 |
| tuned4_val60_seed42 | +0.0075 | +0.0116 | +0.0145 | +0.0120 |
| tuned4_ec_recall_val60_seed42 | +0.0050 | +0.0088 | +0.0145 | +0.0077 |
| hires_bpjdet | +0.2513 | +0.0148 | -0.3478 | +0.4754 |

Interpretation:

1. Among the 398-head comparable runs, `tuned4_val60_seed42` remains best overall (highest accuracy, macro F1, MCC, and kappa).
2. `tuned4_ec_recall_val60_seed42` is close, with similar EC recall gain over tuned3 but slightly lower overall macro F1.
3. Early pilot runs (`tuned`, `tuned2`, `hires_bpjdet`) are not robust comparison baselines for model selection due to tiny overlap (`n=12`).

### Journal entry: expanded statistical test battery (new)

Why:

- We wanted to run as many meaningful evaluation statistics as possible across current tuned variants, beyond basic accuracy/F1.

What was run:

1. Core classification metrics
- accuracy, error rate, balanced accuracy
- macro/weighted/micro precision, recall, F1
- per-class precision/recall/F1/support for `EC`, `IFT`, `OFT`
- confusion matrices (`EC/IFT/OFT` order)
- multiclass MCC and Cohen kappa

2. Probabilistic / calibration metrics
- multiclass log loss
- multiclass Brier score
- ECE (10 bins)
- top-2 accuracy
- class prevalence and predicted class distribution

3. Uncertainty intervals
- 95% Wilson CI for accuracy
- 95% bootstrap CI (2000 resamples, seed `20261012`) for accuracy and macro F1

4. Pairwise significance tests (on the 398-head comparable runs)
- exact McNemar test on disagreement counts
- continuity-corrected McNemar chi-square
- paired bootstrap CI for deltas (accuracy and macro F1)
- permutation test p-value (10000 shuffles) for delta macro F1
- Cohen’s h effect size for accuracy deltas

Artifacts written:

- `outputs/metrics/full_eval_20261012.json`
- `outputs/metrics/full_eval_20261012.md`

Key model-level summary (same evaluated labels as above):

| model | n | accuracy | macro_f1 | mcc | log_loss | ece_10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tuned | 12 | 0.8333 | 0.4667 | 0.5222 | 0.4591 | 0.2989 |
| tuned2 | 12 | 0.8333 | 0.4667 | 0.5222 | 0.4591 | 0.2989 |
| tuned3_val60_seed42 | 398 | 0.7487 | 0.6518 | 0.3756 | 0.8151 | 0.1184 |
| tuned4_val60_seed42 | 398 | 0.7563 | 0.6634 | 0.4006 | 0.8150 | 0.1259 |
| tuned4_ec_recall_val60_seed42 | 398 | 0.7538 | 0.6606 | 0.3908 | 0.8151 | 0.1234 |
| hires_bpjdet | 12 | 1.0000 | 0.6667 | 1.0000 | 0.4591 | 0.3620 |

Pairwise statistical tests (398-head comparable runs):

| pair | mcnemar_exact_p | permutation_p_macro_f1 | delta_accuracy | delta_macro_f1 |
| --- | ---: | ---: | ---: | ---: |
| tuned3 vs tuned4 | 0.2500 | 0.1868 | -0.0075 | -0.0116 |
| tuned3 vs tuned4_ec_recall | 0.6250 | 0.2499 | -0.0050 | -0.0088 |
| tuned4 vs tuned4_ec_recall | 1.0000 | 0.5038 | +0.0025 | +0.0028 |

Interpretation:

1. Directionally, tuned4 remains best on the 398-head set (small gains over tuned3 in accuracy/macro-F1/MCC).
2. The gain magnitude is small; pairwise significance tests do not show strong evidence at conventional thresholds on this sample size.
3. Tuned4 and tuned4_ec_recall are statistically very close; tuned4 has a slight practical edge in aggregate metrics.

Method caveat:

- Pilot models with `n=12` overlap (`tuned`, `tuned2`, `hires_bpjdet`) are included for completeness but should not be used as decisive evidence against the 398-head validation runs.

### Journal entry: baseline 0.5/0.5 replay on random-60 stored probabilities (new)

Why:

- We wanted to compare current tuned logic against the original single-threshold decision rule (`ec=0.5`, `inout=0.5`) on the exact same labeled 398-head subset.

Sources:

- Probabilities: `outputs_hires_bpjdet_tuned3_val60_seed42/predictions.jsonl`
- Human labels: `outputs_hires_bpjdet_tuned3_val60_seed42/review/head_level_review_annotated.csv`

Replay rule:

1. Predict `EC` if `p_EC >= 0.5`
2. Else predict `IFT` if `inout >= 0.5`
3. Else predict `OFT`

Where:

- `p_EC = class_probs['EC']`
- `inout = class_probs['IFT'] / (1 - p_EC)` when `1 - p_EC > 0`

Evaluation set:

- Matched labeled heads: 398 (same keying protocol as tuned3/tuned4 comparisons)

Summary vs tuned3 and tuned4 on the same 398 keys:

| model | n | accuracy | macro_f1 | mcc | log_loss | ece_10 | balanced_acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| replay_0505 | 398 | 0.7437 | 0.6638 | 0.5388 | 0.8151 | 0.1039 | 0.6685 |
| tuned3_val60_seed42 | 398 | 0.7487 | 0.6518 | 0.3756 | 0.8151 | 0.1184 | 0.6556 |
| tuned4_val60_seed42 | 398 | 0.7563 | 0.6634 | 0.4006 | 0.8150 | 0.1259 | 0.6684 |

Replay per-class precision/recall/F1:

| class | precision | recall | f1 |
| --- | ---: | ---: | ---: |
| EC | 0.5789 | 0.4783 | 0.5238 |
| IFT | 0.8650 | 0.8367 | 0.8506 |
| OFT | 0.5577 | 0.6905 | 0.6170 |

Replay confusion matrix (rows=true, cols=pred, class order `EC/IFT/OFT`):

- `[[33, 14, 22], [16, 205, 24], [8, 18, 58]]`

Delta against tuned variants:

1. replay vs tuned3
- `delta_accuracy = -0.0050`
- `delta_macro_f1 = +0.0120`

2. replay vs tuned4
- `delta_accuracy = -0.0126`
- `delta_macro_f1 = +0.0004`

Interpretation:

1. The original 0.5/0.5 replay is competitive in macro-F1 on this subset (slightly above tuned3 and nearly tied with tuned4), but lower in raw accuracy than both tuned3 and tuned4.
2. Tuned4 still provides the best overall accuracy among compared variants on this 398-head set.
3. The replay result reinforces that threshold policy choice changes class tradeoffs (especially EC recall vs precision balance), so model-selection should continue to prioritize the full metric/test battery and not a single scalar.
