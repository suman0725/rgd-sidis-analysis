#!/bin/bash
# ─────────────────────────────────────────────────────────────────
#  run_root2parquet.sh
#  Batch ROOT → parquet for a single target/run directory
#
#  Usage:
#    bash run_root2parquet.sh <TARGET> <RUN>
#
#  Examples:
#    bash run_root2parquet.sh LD2 018420
#    bash run_root2parquet.sh CxC 018454
# ─────────────────────────────────────────────────────────────────

TARGET=${1}
RUN=${2}
MAX_JOBS=8    # parquet jobs are CPU/memory heavy — keep lower than ROOT jobs

ROOT_DIR="/volatile/clas12/suman/00_RGD_Analysis/data/experimental/root/${TARGET}/${RUN}"
OUT_DIR="/volatile/clas12/suman/00_RGD_Analysis/data/experimental/parquet/${TARGET}/${RUN}"
LOG_DIR="/volatile/clas12/suman/00_RGD_Analysis/data/experimental/parquet/logs"
SCRIPT="/work/clas12/suman/00_RGD_Analysis/SIDIS_Analysis_Python/scripts/root_2_parquet.py"

if [[ -z "$TARGET" || -z "$RUN" ]]; then
    echo "Usage: bash run_root2parquet.sh <TARGET> <RUN>"
    exit 1
fi

if [[ ! -d "$ROOT_DIR" ]]; then
    echo "ERROR: ROOT directory not found: ${ROOT_DIR}"
    exit 1
fi

mkdir -p "${OUT_DIR}" "${LOG_DIR}"

mapfile -t ROOT_FILES < <(ls "${ROOT_DIR}"/*.root 2>/dev/null | sort)
TOTAL=${#ROOT_FILES[@]}

if [[ $TOTAL -eq 0 ]]; then
    echo "ERROR: No ROOT files found in ${ROOT_DIR}"
    exit 1
fi

echo "========================================"
echo "  Target  : ${TARGET}"
echo "  Run     : ${RUN}"
echo "  Files   : ${TOTAL}"
echo "  Output  : ${OUT_DIR}"
echo "  Started : $(date)"
echo "========================================"

convert_one() {
    local root_file="$1"
    local base
    base="$(basename "${root_file}" .root)"
    local out_parquet="${OUT_DIR}/sidis_${TARGET}_OB_${base}.root.parquet"

    # skip if already done
    if [[ -f "${out_parquet}" ]]; then
        echo "[SKIP] ${base}"
        return 0
    fi

    python3 "${SCRIPT}" \
        --target "${TARGET}" \
        --root-file "${root_file}" \
        --out-dir "${OUT_DIR}" \
        > "${LOG_DIR}/${TARGET}_${base}.log" 2>&1

    if [[ $? -eq 0 ]]; then
        echo "[DONE] ${base}"
    else
        echo "[ERROR] ${base} — check ${LOG_DIR}/${TARGET}_${base}.log"
    fi
}

export -f convert_one
export TARGET OUT_DIR LOG_DIR SCRIPT

for f in "${ROOT_FILES[@]}"; do
    convert_one "$f" &
    while (( $(jobs -r | wc -l) >= MAX_JOBS )); do
        sleep 5
    done
done

wait

echo ""
echo "========================================"
echo "  FINISHED: ${TARGET} / ${RUN}"
echo "  Parquet files: $(ls "${OUT_DIR}"/*.parquet 2>/dev/null | wc -l) / ${TOTAL}"
echo "  Ended : $(date)"
echo "========================================"
