#!/bin/bash
# S5 assembly after run_e06.sh: build every candidate submission, profile each against the current best (S4, 0.969),
# validate. Usage: bash s5.sh <france_frac from calib.py loco>
set -eo pipefail
cd "$(dirname "$0")/../src"
export PYTHONIOENCODING=utf-8 ER_WORK="${ER_WORK:-$(pwd)/../work_e06}" ER_DATA="${ER_DATA:-D:/Amazon ML/student_resource/dataset}"
PY="${PY:-python}"
export STAGES="${STAGES:-1}"
R="D:/Amazon ML"
S4="$R/output_S4_consensus/matching_results.tsv"
FRAC="${1:?france fraction}"
O="$R/s5"
mkdir -p "$O"/{e06,e06fr,maj,all3}
{
cp ../output_e06/*.tsv "$O/e06/"
"$PY" calib.py test "$O/e06fr" france "$FRAC"
# 3-model votes: E02, S3 (legacy candidate set) + E06-France-calibrated (own candidate set) -> union candidates
"$PY" vote.py "$O/maj/matching_results.tsv" 2 "$O/e06fr/candidate_pairs.tsv,$R/output_S3/candidate_pairs.tsv" \
      "$O/e06fr/matching_results.tsv" "$R/output_E02/matching_results.tsv" "$R/output_S3/matching_results.tsv"
"$PY" vote.py "$O/all3/matching_results.tsv" 3 "$O/e06fr/candidate_pairs.tsv,$R/output_S3/candidate_pairs.tsv" \
      "$O/e06fr/matching_results.tsv" "$R/output_E02/matching_results.tsv" "$R/output_S3/matching_results.tsv"
for v in e06 e06fr maj all3; do
  echo "================ $v vs S4"
  "$PY" "$R/student_resource/utils/validate_submission.py" --matching "$O/$v/matching_results.tsv" --candidate "$O/$v/candidate_pairs.tsv" --test-dir "$ER_DATA/test" --check-ids | tail -2
  "$PY" diffprof.py "$S4" "$O/$v/matching_results.tsv" | sed -n '/ -> /,$p'
done
} 2>&1 | tee -a "$O/s5.log"
