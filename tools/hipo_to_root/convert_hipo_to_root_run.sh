#!/bin/bash

##############################################
# HIPO → ROOT converter for RGD data
# Features: Indexed Selection, Summary Reporting, and Smart Skipping
##############################################

# --------- CONFIGURATION ---------
BASE_INPUT="/cache/clas12/rg-d/production/pass1/recon"
BASE_OUTPUT="/volatile/clas12/suman/00_RGD_Analysis/data/experimental/root"
MACRO_PATH="/w/hallb-scshelf2102/clas12/suman/03_tools/hipo/c12_working/hipo_to_root.cxx"
MAX_JOBS=30
# -------------------------------

if [ -z "$BASH_VERSION" ]; then
    echo "Please run this script with bash."
    exit 1
fi

mkdir -p "${BASE_OUTPUT}"

##############################################
# 1. Choose TARGET
##############################################
echo "Available targets under ${BASE_INPUT}:"
ls "${BASE_INPUT}"
echo

read -rp "Enter target name (e.g., CuSn, CxC, LD2): " TARGET

INPUT_TARGET_DIR="${BASE_INPUT}/${TARGET}"
OUTPUT_TARGET_DIR="${BASE_OUTPUT}/${TARGET}"

if [[ ! -d "${INPUT_TARGET_DIR}" ]]; then
    echo "ERROR: Input target directory does not exist: ${INPUT_TARGET_DIR}"
    exit 1
fi

mkdir -p "${OUTPUT_TARGET_DIR}"
LOG_FILE="${OUTPUT_TARGET_DIR}/processed_${TARGET}.txt"
touch "${LOG_FILE}"

##############################################
# 2. Choose RUN(S)
##############################################
RUN_ROOT="${INPUT_TARGET_DIR}/dst/recon"
mapfile -t ALL_RUN_DIRS < <(ls -1 "${RUN_ROOT}" | sort)

echo "------------------------------------------------"
echo "Found ${#ALL_RUN_DIRS[@]} run directories for $TARGET."
echo "Choose Run Selection Mode:"
echo "  1) Single Run (Interactive file selection)"
echo "  2) Multiple Runs by INDEX (e.g., 1 5 10-15)"
echo "  3) ALL Runs (Show summary first)"
echo "------------------------------------------------"
read -rp "Choice [1-3]: " RUN_MODE

SELECTED_RUNS=()

if [[ "$RUN_MODE" == "1" || "$RUN_MODE" == "2" ]]; then
    echo "Available Runs:"
    for i in "${!ALL_RUN_DIRS[@]}"; do
        printf "  [%3d] %s" "$((i+1))" "${ALL_RUN_DIRS[$i]}"
        [[ $(( (i+1) % 4 )) -eq 0 ]] && echo ""
    done
    echo -e "\n"

    if [[ "$RUN_MODE" == "1" ]]; then
        read -rp "Enter Index of the run: " R_IDX
        SELECTED_RUNS+=("${ALL_RUN_DIRS[$((R_IDX-1))]}")
    else
        read -rp "Enter indices/ranges (e.g., 1 2 5-10): " R_INPUT
        for item in $R_INPUT; do
            if [[ $item =~ ^([0-9]+)-([0-9]+)$ ]]; then
                for (( j=${BASH_REMATCH[1]}; j<=${BASH_REMATCH[2]}; j++ )); do
                    SELECTED_RUNS+=("${ALL_RUN_DIRS[$((j-1))]}")
                done
            else
                SELECTED_RUNS+=("${ALL_RUN_DIRS[$((item-1))]}")
            fi
        done
    fi
elif [[ "$RUN_MODE" == "3" ]]; then
    SELECTED_RUNS=("${ALL_RUN_DIRS[@]}")
fi

##############################################
# 3. Collect Files & Show Summary
##############################################
FILES_TO_PROCESS=()
declare -A FILE_TO_RUN 
TOTAL_HIPO_COUNT=0

echo -e "\n--- SELECTION SUMMARY ---"
printf "%-10s | %-15s | %-10s\n" "Index" "Run Number" "File Count"
echo "------------------------------------------------"

for RUN_NUM in "${SELECTED_RUNS[@]}"; do
    RUN_DIR="${RUN_ROOT}/${RUN_NUM}"
    mapfile -t HIPO_FILES < <(ls "${RUN_DIR}"/rec_clas_*.hipo 2>/dev/null | sort || true)
    COUNT=${#HIPO_FILES[@]}
    
    IDX="?"
    for i in "${!ALL_RUN_DIRS[@]}"; do
        [[ "${ALL_RUN_DIRS[$i]}" == "$RUN_NUM" ]] && IDX=$((i+1)) && break
    done

    printf "[%3d]      | %-15s | %-10d\n" "$IDX" "$RUN_NUM" "$COUNT"

    if [[ "$RUN_MODE" == "1" && $COUNT -gt 0 ]]; then
        echo -e "\nFiles in $RUN_NUM:"
        for i in "${!HIPO_FILES[@]}"; do
            printf "  [%3d] %s\n" "$((i+1))" "$(basename "${HIPO_FILES[$i]}")"
        done
        read -rp "Select: 1) Single 2) First N 3) All 4) Range: " F_MODE
        case $F_MODE in
            1) read -rp "Idx: " I; TEMP_LIST=("${HIPO_FILES[$((I-1))]}") ;;
            2) read -rp "N: " N; TEMP_LIST=("${HIPO_FILES[@]:0:N}") ;;
            3) TEMP_LIST=("${HIPO_FILES[@]}") ;;
            4) read -rp "Range (S-E): " R; [[ $R =~ ^([0-9]+)-([0-9]+)$ ]]; S=${BASH_REMATCH[1]}; E=${BASH_REMATCH[2]}
               TEMP_LIST=("${HIPO_FILES[@]:$((S-1)):$((E-S+1))}") ;;
        esac
    else
        TEMP_LIST=("${HIPO_FILES[@]}")
    fi

    for F in "${TEMP_LIST[@]}"; do
        FILES_TO_PROCESS+=("$F")
        FILE_TO_RUN["$F"]="$RUN_NUM"
        ((TOTAL_HIPO_COUNT++))
    done
done

echo "------------------------------------------------"
echo "TOTAL FILES QUEUED: $TOTAL_HIPO_COUNT"
echo "------------------------------------------------"
read -rp "Proceed with conversion? (y/n): " CONFIRM
[[ "$CONFIRM" != "y" ]] && exit 0

##############################################
# 4. Smart Processing Logic
##############################################
process_file() {
    local in_file="$1"
    local run_num="${FILE_TO_RUN[$in_file]}"
    local base_name="$(basename "${in_file}" .hipo)"
    
    # Define paths for the message
    local out_dir="${OUTPUT_TARGET_DIR}/${run_num}"
    local out_file="${out_dir}/${base_name}.root"

    # 1. Check Log Check
    if grep -q "${in_file}" "${LOG_FILE}"; then
        echo "[SKIP - LOGGED] Run:${run_num} | File:${base_name} | Location: ${out_dir}"
        return 0
    fi

    # 2. Physical Disk Check
    if [[ -f "${out_file}" ]]; then
        echo "[SKIP - EXISTS] Run:${run_num} | File:${base_name} | Location: ${out_dir}"
        # Sync log if missing
        echo "${in_file}" >> "${LOG_FILE}"
        return 0
    fi

    mkdir -p "${out_dir}"
    local macro_dir=$(dirname "${MACRO_PATH}")
    local macro_name=$(basename "${MACRO_PATH}")

    # Use a subshell to run the conversion
    (
        cd "${macro_dir}" || exit
        root -b -q "${macro_name}(\"${in_file}\",\"${out_file}\")" > /dev/null 2>&1
    )

    if [[ $? -eq 0 ]]; then
        echo "${in_file}" >> "${LOG_FILE}"
        echo "[DONE] Run:${run_num} | Created: ${base_name}.root in ${out_dir}"
    else
        echo "[ERROR] Run:${run_num} | Failed: ${base_name} | Check Macro: ${MACRO_PATH}"
    fi
}
##############################################
# 5. Execution
##############################################
START_TIME=$(date +%s)
echo "Starting conversion (Parallel: $MAX_JOBS)..."

for f in "${FILES_TO_PROCESS[@]}"; do
    process_file "${f}" &
    while (( $(jobs -r | wc -l) >= MAX_JOBS )); do
        sleep 2
    done
done

wait
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo "------------------------------------------------"
echo "Workflow Complete!"
echo "Total Time: $((ELAPSED/3600))h $(((ELAPSED%3600)/60))m $((ELAPSED%60))s"
echo "------------------------------------------------"
