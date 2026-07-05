#!/usr/bin/env bash
set -euo pipefail

PROJ=${PROJ:-/gemini/code/LightRecon3D}
PYTHON=${PYTHON:-/root/miniconda3/envs/lightrecon/bin/python}
ROOT=${ROOT:-/gemini/data-1/Structured3D}
RUNROOT=${RUNROOT:-/gemini/data-1/lightrecon_runs}
OUT_ROOT=${OUT_ROOT:-$RUNROOT/stage3_dualview_val_groups_v1}
NUM_GROUPS=${NUM_GROUPS:-5}
MIN_PAIRS=${MIN_PAIRS:-10}
START_GROUP=${START_GROUP:-0}
PAIR_STRATEGY=${PAIR_STRATEGY:-all}

export PROJ PYTHON ROOT RUNROOT
export PYTHONPATH="$PROJ:${PYTHONPATH:-}"

cd "$PROJ"
mkdir -p "$OUT_ROOT/logs"
GROUPS_TSV="$OUT_ROOT/selected_groups.tsv"
GROUPS_RAW="$OUT_ROOT/selected_groups.raw.log"

cat > "$OUT_ROOT/select_groups.py" <<'PY'
import contextlib
import os
import sys
from collections import defaultdict
from dataloaders.s3d_dataset import Structured3DDataset

root = os.environ["ROOT"]
num_groups = int(os.environ.get("NUM_GROUPS", "5"))
min_pairs = int(os.environ.get("MIN_PAIRS", "10"))
start_group = int(os.environ.get("START_GROUP", "0"))
pair_strategy = os.environ.get("PAIR_STRATEGY", "all")

with contextlib.redirect_stdout(sys.stderr):
    ds = Structured3DDataset(
        root_dir=root,
        split="val",
        train_ratio=0.9,
        image_size=(512, 512),
        input_mode="pair",
        pair_strategy=pair_strategy,
    )

groups = defaultdict(list)
for idx, sample in enumerate(ds.samples):
    groups[sample["pair_group"]].append(idx)

rows = sorted(
    [(len(indices), group, indices) for group, indices in groups.items() if len(indices) >= min_pairs],
    key=lambda row: (-row[0], row[1]),
)
rows = rows[start_group:start_group + num_groups]
for ordinal, (count, group, indices) in enumerate(rows):
    print(f"{ordinal}\t{count}\t{','.join(map(str, indices[:min_pairs]))}\t{group}")
PY

NUM_GROUPS="$NUM_GROUPS" MIN_PAIRS="$MIN_PAIRS" START_GROUP="$START_GROUP" PAIR_STRATEGY="$PAIR_STRATEGY" \
  "$PYTHON" "$OUT_ROOT/select_groups.py" 2>&1 | tee "$GROUPS_RAW"
awk -F'\t' '$1 ~ /^[0-9]+$/ && $2 ~ /^[0-9]+$/ {print}' "$GROUPS_RAW" > "$GROUPS_TSV"
echo "Selected groups:"
cat "$GROUPS_TSV"

if [ ! -s "$GROUPS_TSV" ]; then
  echo "No groups selected. Check ROOT=$ROOT MIN_PAIRS=$MIN_PAIRS" >&2
  exit 1
fi

while IFS=$'\t' read -r ordinal pair_count indices group_path; do
  group_name=$(printf "group_%03d_pairs_%s" "$ordinal" "$pair_count")
  debug_root="$OUT_ROOT/$group_name"
  echo "[group $ordinal] pair_count=$pair_count indices=$indices group=$group_path"
  DEBUG_ROOT="$debug_root" INDICES="$indices" ./run_stage3_dualview_onegroup_v1.sh \
    2>&1 | tee "$OUT_ROOT/logs/${group_name}.log"
done < "$GROUPS_TSV"

"$PYTHON" summarize_stage3_val_groups.py \
  --root "$OUT_ROOT" \
  --output_csv "$OUT_ROOT/stage3_val_groups_summary.csv" \
  --output_json "$OUT_ROOT/stage3_val_groups_summary.json"

echo "Done. Summary:"
echo "$OUT_ROOT/stage3_val_groups_summary.csv"
echo "$OUT_ROOT/stage3_val_groups_summary.json"