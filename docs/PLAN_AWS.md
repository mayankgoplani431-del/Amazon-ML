# Plan: moving training to AWS (written Sep 26, ~22:15 IST; deadline Sep 27 23:59 IST, ~25 h left)

Current best: **S5 = E02 ∩ S3 ∩ E06 = 0.971**. E07 (mined normalization fixes) is running on the laptop and is the
baseline that AWS has to beat. **Everything AWS adds is incremental on top of E07.**

## 1. Does AWS help? Yes, but only a little. It removes memory cuts; it does not fix the train/test gap

The gap between validation and the leaderboard (0.980 OOF vs 0.971 LB) comes from test density and the unseen country
(docs/audit.md, audit_s5.md). More RAM does not fix that. What AWS does fix is the list of things we cut on the laptop.

| # | Item | Type | Evidence | OOF effect | **Expected LB** |
|---|---|---|---|---|---|
| A1 | Stage-2 re-scorer back on (`STAGES=2`) | Unblocks something known to help | E01 0.9679→0.9684; E03a ≈ equal | +0.0005 | **0 to +0.001** |
| A2 | Stage-1 on 100% of S1 groups (now 30%), 1500 rounds | Unblocks something known to help | PLAN_98 estimate +0.001–0.002 together with A1 | +0.0003–0.001 | **0 to +0.001** |
| A3 | Stage-0 pruner sample 5%→15% | Unblocks | top-20 already keeps 99.86% of union true pairs (7,448,317 / 7,458,509) | ~0 | **~0** (free, do it) |
| A4 | LOCO calibration for France (`calib.py loco`, both directions) | Unblocks, but the sign is uncertain | L01 proxy: unseen country FP ×3. But S5 France profile is already close to in-domain (5.38% empty vs 5.9%) | proxy only | **−0.001 to +0.002** |
| A5 | 3 seeds + XGBoost (`hist`, Apache-2.0), **probability-averaged** | Speculative | Same features and candidates, so errors stay highly correlated. Gain comes from smoother probabilities in expected-F0.5, not from voting | +0.0003–0.001 | **0 to +0.001** |
| A6 | Stronger E08 replaces E06 in the vote (E02 ∩ S3 ∩ E08) | Known pattern, nearly used up | audit_s5 §4: removed-set precision f fell from 0.81 to 0.43; break-even is about 0.23 | n/a | **0 to +0.001** |
| A7 | Blocking K ×2 (comb 50→100, addr 25→50, name_ch 12→25, rev 5→10) | Speculative and expensive | Dev sample: comb alone 0.922→0.935 (blk_exp3). The union gain is smaller. On test, recall additions look distractor-like (E06 added set: US number mismatch 0.667) | +0.001–0.003 | **−0.001 to +0.002** |
| A8 | Stacking meta-learner over E02/S3/E0x | Speculative, risky | It learns to trust S3 on train, which is the S3 failure again (S3: +0.003 OOF, −0.008 LB) | + on OOF | **−0.003 to +0.001**, skip unless density-stressed |
| A9 | Stage-0 top_n 20→30 | Skip | +3.3k of 7.64M true pairs (0.04%) for 50% more pairs | ~0 | ~0 |

**Combined realistic range: 0.972–0.975 (central ≈0.973).** The items overlap, so the gains do not add up.
The chance of reaching ≥0.98 is under 5%. The most reliable parts are A1+A2+A3: one "E08 = E07 at full size" model
used as the third voter.

## 2. Instance choice: memory-optimized CPU, not GPU

Where the time goes: sparse TF-IDF top-K (`sparse_dot_topn`, CPU-only), rapidfuzz `cpdist` (CPU), polars joins, and
LightGBM on 13–44M rows × ~70 features (44M × 70 × 4 B ≈ 12 GB dense, plus polars copies at peak ≈ 40–60 GB with
stage 2). A GPU would speed up only the GBM fit (≈15% of wall time). GPU boxes also have less RAM per dollar, and
the LightGBM GPU build (OpenCL) is fragile. **Choose r7i/r6i.**

| Instance | vCPU / RAM | On-demand (approx, verify in the console) | Use |
|---|---|---|---|
| **r7i.16xlarge** | 64 / 512 GB | ~$4.2/h | Box A: E08 + A7 (larger K) |
| **r7i.8xlarge** | 32 / 256 GB | ~$2.1/h | Box B: E08 at the E07 blocking + A4 + A5 |
| g5.8xlarge | 32 / 128 GB, A10G | ~$2.4/h | Not worth it: only speeds up the fit, less RAM |
| p3.2xlarge | 8 / 61 GB, V100 | ~$3.1/h | No: too little RAM and too few CPUs |

**Cost:** A ≈10 h × $4.2 + B ≈12 h × $2.1 + 2 × 400 GB gp3 + S3 + egress ≈ **$70–90**. With re-runs the ceiling is
~$150, inside the $200 credits. Use **on-demand**. The whole job fits easily in the budget, and a spot interruption
costs hours we don't have. Spot is only for optional extra seeds on a third box, and only if its separate quota
exists.

## 3. Setup (~1–1.5 h, done in parallel with the laptop run)

**Do not edit the laptop's live `src/`.** `run_e06.sh` starts a new Python process for every step, so an edit would
leak into the running E07. Copy the repo and make all edits in the copy.

1. **Quota first (the biggest blocker).** Service Quotas → EC2 → "Running On-Demand Standard (A,C,D,H,I,M,R,T,Z)
   instances" must be ≥ 96 vCPU (both boxes) or ≥ 32 (Box B only). New accounts are often at 5–32. Request the
   increase immediately; approval can take hours.
2. Region ap-south-1 (Mumbai). Ubuntu 24.04. Root gp3 volume 400 GB. SSH only from our IP. Private S3 bucket with
   Block Public Access on (fair-play: the data must never be public).
3. Data: zip `student_resource/dataset` (~3 GB raw, ~1 GB zipped), then `scp` it (or `aws s3 cp` to the private
   bucket). Also upload `output_E02/`, `output_S3/` and later the E07 outputs, so that `vote.py` runs on AWS and
   not on the laptop.
4. Environment: `uv venv -p 3.11 && uv pip install -r requirements.txt xgboost==<pin>`. Pin the version and add it
   to requirements.txt (Apache-2.0). Smoke test with the stdlib demos: `python vote.py`, `python calib.py`.
5. Code edits (small, driven by environment variables so the laptop behaviour doesn't change):
   - Thread counts are hard-coded to 14: `pipeline.P1` / `predict_stage`, `stage0.PARAMS` / `predict`,
     `blocking.N_THREADS`, `calib.py:64`. Change them to `int(os.environ.get('NTHREADS', 14))`. Without this, a
     64-vCPU box runs no faster than the laptop.
   - `pipeline.train_stage`: `sample_frac` from `S1_FRAC` (default 0.3), `rounds` from `ROUNDS`, `P1['seed']` from
     `SEED`. Keep `fold_of` seed 7 fixed so every model shares the same OOF folds.
   - `pipeline.train_stage` / `predict_stage`: `MODEL=xgb` branch (`xgb.train`, `tree_method='hist'`,
     `max_leaves=255`, `grow_policy='lossguide'`, same weights) (~15 lines).
   - `scripts/run_e06.sh`: `export STAGES=${STAGES:-1}` (it is hard-coded to 1 now). `ER_DATA` is passed explicitly.
   - `candidates.CFG`: set the K values from a `BLOCK_K2=1` flag (Box A only).
   - New `src/blend.py` (~50 lines): read `oof_stage*.parquet` / `test_stage*.parquet` from several work dirs
     (same candidate set, same `qi, pi`), take the mean of the logits, run `run_train.sweep` with the
     density-stressed `mk` (owned, alpha=1.9), apply the rule to test, and write the files via `run_test.write_lists`.
6. Run each box in `tmux`:
   `ER_DATA=~/data ER_WORK=~/work_e08 OUT=~/output_e08 EXP=E08 STAGES=2 STAGE0_FRAC=0.15 S1_FRAC=1.0 ROUNDS=1500 NTHREADS=60 bash scripts/run_e06.sh`
   Then run `LOCO_TR=300 LOCO_EV=1000 python calib.py loco us india` and `... india us`.
7. Retrieve only the small files (matching/candidate TSVs gzipped, `test_stage*.parquet`, `oof_*.parquet`, models,
   logs). Terminate both boxes and delete the EBS volumes and the S3 bucket after the package is built.

## 4. Modelling plan and how each result is judged

OOF alone is not trusted (S3 lesson). Every candidate submission must pass all four gates:

1. Official validator `--check-ids` and `check_output.py`.
2. OOF macro F0.5 **plain and density-stressed** (alpha 1.9) do not drop on either fold.
3. `diffprof.py S5 <new>`: per-country first-house-number mismatch of the **added** and **removed** pairs, compared
   with the references in audit_s5 §2. Break-even under F0.5 (≈+0.06 per true pair added, ≈−0.20 per false pair):
   **an added set helps only if it is ≳77% precise**, so its mismatch rate must sit near the true-pair reference
   (US 0.11, India 0.21), not near the false-positive reference (US 0.35, India 0.37). **A removed set helps only if
   f ≳ 0.23.**
4. Per-country profile stays in line with the others (empty rate ~5.8–5.9%, ~3.3 matches/S1). France moves only
   if LOCO says so.

| Box | Run | Output |
|---|---|---|
| B | E08 = E07 code, same blocking, A1+A2+A3, seed 0 | Main new voter |
| B | +2 LightGBM seeds, +1 XGBoost (stage 1 only, reuses the cached features) → `blend.py` | Blend of E08 models |
| B | LOCO both directions. Apply France δ (`calib.py test`) **only if the label-free pick improves proxy F in both directions** | France-calibrated variant |
| A | E08-K2 = E08 + A7. Keep it only if blocking recall rises ≥0.2 pt **and** its added-pair profile passes gate 3 | Recall variant |

## 5. Timeline (IST)

| When | What |
|---|---|
| Sep 26 22:30–00:00 | Quota request, launch, upload, environment, code edits in the copy, smoke test. **Gate: quota approved by ~01:00, otherwise NO-GO** |
| 00:00 | Start Box B (E08), then Box A (E08-K2) |
| ~05:30 / ~08:30 | B / A finish train + test (estimate: laptop cycle ~5 h at 14 threads; bigger settings, more cores) |
| 05:30–09:00 | Seeds/XGB + LOCO on B; blend; diffprof vs S5 |
| Sep 27 (5 submissions) | **#1** E02 ∩ S3 ∩ E07 (laptop, if planned) · **#2** E02 ∩ S3 ∩ E08(blend) · **#3** #2 + France δ (only if LOCO passed) · **#4** best of E08-K2 / majority variant · **#5 reserved** for the final or an emergency |
| **18:00 model freeze** | No new training after this point |
| 18:00–21:30 | `scripts/run_final.sh` (exact env vars for every model + the vote). Update `make_package.py` (it bundles only `run_e06.sh`; add `blend.py`, the new scripts and the xgboost pin). Documentation template states the hardware (needs >16 GB RAM: tested on r7i). Run both validators on the packaged output |
| 21:30–23:59 | Buffer; upload the package; terminate AWS |

## 6. Risks and go/no-go

| Risk | Mitigation |
|---|---|
| vCPU quota not approved in time | Hard gate at ~01:00; fall back to laptop-only |
| Gains land inside public-LB noise (~±0.001) | Submit only if gates 1–4 pass; never pick by LB alone |
| Stage 2 / stacking overfits train density | Density-stressed sweep; no stacking over S3 |
| Final package cannot reproduce AWS models | One `run_final.sh` with pinned requirements; the doc states the RAM needed |
| Laptop E07 corrupted by code edits | Edit only the AWS copy |
| Credits | On-demand ~$70–90, ceiling ~$150; set a billing alarm at $150 |

**Recommendation: conditional GO, time-boxed.** Go if the quota is approved by ~01:00 IST and the data upload works.
Expect **+0.001 to +0.004 (0.972–0.975)**, mainly from a full-size E08 (stage 2, 100% of groups, blended seeds) used
as the third voter, plus possibly France calibration. Do not expect 0.98. **NO-GO** if the quota is not approved in
time: then keep the laptop on E07 plus consensus, and spend the remaining effort on a clean final package.
