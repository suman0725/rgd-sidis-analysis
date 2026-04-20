#!/bin/bash
# ─────────────────────────────────────────────────────────────────
#  run_conversion.sh
#  Non-interactive HIPO → ROOT conversion for a single target/run
#
#  Usage:
#    bash run_conversion.sh <TARGET> <RUN> [MAX_FILES]
#
#  Examples:
#    bash run_conversion.sh LD2 018420        # all files in run
#    bash run_conversion.sh CxC 018454 30     # first 30 files only
# ─────────────────────────────────────────────────────────────────

TARGET=${1}
RUN=${2}
MAX_FILES=${3:-9999}   # default: all files

BASE_INPUT="/cache/clas12/rg-d/production/pass1/recon"
BASE_OUTPUT="/volatile/clas12/suman/00_RGD_Analysis/data/experimental/root"
MACRO="/work/clas12/suman/03_tools/hipo/c12_working/hipo_to_root.cxx"
MAX_JOBS=30

if [[ -z "$TARGET" || -z "$RUN" ]]; then
    echo "Usage: bash run_conversion.sh <TARGET> <RUN> [MAX_FILES]"
    exit 1
fi

INPUT_DIR="${BASE_INPUT}/${TARGET}/dst/recon/${RUN}"
OUTPUT_DIR="${BASE_OUTPUT}/${TARGET}/${RUN}"
LOG_FILE="${BASE_OUTPUT}/${TARGET}/processed_${TARGET}.txt"

mkdir -p "${OUTPUT_DIR}"
touch "${LOG_FILE}"

# collect files (sorted, capped at MAX_FILES)
mapfile -t ALL_FILES < <(ls "${INPUT_DIR}"/rec_clas_*.hipo 2>/dev/null | sort | head -"${MAX_FILES}")

TOTAL=${#ALL_FILES[@]}
if [[ $TOTAL -eq 0 ]]; then
    echo "ERROR: No HIPO files found in ${INPUT_DIR}"
    exit 1
fi

echo "========================================"
echo "  Target  : ${TARGET}"
echo "  Run     : ${RUN}"
echo "  Files   : ${TOTAL}"
echo "  Output  : ${OUTPUT_DIR}"
echo "  Started : $(date)"
echo "========================================"

convert_file() {
    local in_file="$1"
    local base
    base="$(basename "${in_file}" .hipo)"
    local out_file="${OUTPUT_DIR}/${base}.root"

    # skip if already logged or file exists on disk
    if grep -qF "${in_file}" "${LOG_FILE}" 2>/dev/null; then
        echo "[SKIP-LOG ] ${base}"
        return 0
    fi
    if [[ -f "${out_file}" ]]; then
        echo "[SKIP-DISK] ${base}"
        echo "${in_file}" >> "${LOG_FILE}"
        return 0
    fi

    # run conversion
    local macro_dir macro_name
    macro_dir="$(dirname "${MACRO}")"
    macro_name="$(basename "${MACRO}")"
    (
        cd "${macro_dir}" || exit 1
        root -b -q "${macro_name}(\"${in_file}\",\"${out_file}\")" > /dev/null 2>&1
    )

    if [[ $? -eq 0 ]]; then
        echo "${in_file}" >> "${LOG_FILE}"
        echo "[DONE ] ${base}.root"
    else
        echo "[ERROR] ${base} — conversion failed"
    fi
}

export -f convert_file
export OUTPUT_DIR LOG_FILE MACRO

# launch files in parallel, cap at MAX_JOBS
DONE=0
for f in "${ALL_FILES[@]}"; do
    convert_file "$f" &
    (( DONE++ ))
    # throttle
    while (( $(jobs -r | wc -l) >= MAX_JOBS )); do
        sleep 3
    done
done

wait

echo ""
echo "========================================"
echo "  FINISHED: ${TARGET} / ${RUN}"
echo "  Converted: $(ls "${OUTPUT_DIR}"/*.root 2>/dev/null | wc -l) / ${TOTAL} files"
echo "  Ended : $(date)"
echo "========================================"
