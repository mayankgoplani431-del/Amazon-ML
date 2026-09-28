#!/bin/bash
# E06 full cycle (~5 h): fixed normalization (postal codes, dotted legal forms, visible "Holding"),
# new exact-key blocking passes, test-density training (no-owner negatives x1.9, density-sensitive features
# dropped, density-stressed validation), then test scoring + diagnostics + unseen-country calibration check.
# Fresh work dir: never reuses caches built with the old normalization.
set -eo pipefail
cd "$(dirname "$0")/../src"
export PYTHONIOENCODING=utf-8 ER_WORK="${ER_WORK:-$(pwd)/../work_e06}" ER_DATA="${ER_DATA:-D:/Amazon ML/student_resource/dataset}"
PY="${PY:-python}"
OUT="${OUT:-../output_e06}"; EXP="${EXP:-E06}"   # E07: ER_WORK=../work_e07 OUT=../output_e07 EXP=E07
export STAGES=1   # stage 2 crashed twice on memory (16 GB laptop); stage 1 = 0.9799 OOF
export DROP=cvc_name,cvc_addr,g_same_addr,g_same_name,n_freq_q,n_freq_c,g_size,s0_q_n,s0_p_n,s0_q_sum
mkdir -p "$ER_WORK"
{
"$PY" prep.py train test
[ -f "$ER_WORK/cache/cand_train.done" ] || { "$PY" candidates.py train || "$PY" candidates.py train; touch "$ER_WORK/cache/cand_train.done"; }   # retry once (intermittent native segfault); skip when done
[ -f "$ER_WORK/models/decision_stage1.json" ] || NEG_W=1.9 ALPHA=1.9 "$PY" run_train.py "$EXP" 20 || NEG_W=1.9 ALPHA=1.9 "$PY" run_train.py "$EXP" 20   # retry resumes from saved parts/models
[ -f "$ER_WORK/cache/cand_test.done" ] || { "$PY" candidates.py test || "$PY" candidates.py test; touch "$ER_WORK/cache/cand_test.done"; }
"$PY" run_test.py "$OUT" 20
"$PY" check_output.py "$OUT" "$ER_DATA/test"
"$PY" diag.py "$OUT/matching_results.tsv" "$EXP"
# calib.py loco us india needs >16 GB free commit on this laptop: run separately when memory allows
} 2>&1 | tee -a "$ER_WORK/run_e06.log"
