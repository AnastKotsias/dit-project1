#!/usr/bin/env bash
# runs ./verify for the experimental study, one output and one log file per run
# usage: scripts/verify_experiments.sh validation|validation-grid|test
#   validation       one verification parameter changed at a time
#   validation-grid  frozen configuration, every method, D and C, on validation
#   test             the same grid on test
# every variable below can be set from the environment, e.g. VALIDATION_D=1000
# DRY_RUN=1 prints the commands without running them
set -euo pipefail

phase=${1:-}
if [[ $phase != validation && $phase != validation-grid && $phase != test ]]; then
    echo "usage: $0 validation|validation-grid|test" >&2
    exit 1
fi
# paths relative to the repository root
cd "$(dirname "$0")/.."

# data and BoVW options of assignment 1
HP=${HP:-data/hpatches-sequences-release}
MIR=${MIR:-data/mirflickr}
SPLIT=${SPLIT:-split.txt}
VOCAB=${VOCAB:-256}
S=${S:-500}
SEED=${SEED:-1}
OUT_DIR=${OUT_DIR:-results/verify}
# -cache of vocabulary and BoVW vectors, shared by all runs, empty for no cache
CACHE_DIR=${CACHE_DIR-results/cache}
DRY_RUN=${DRY_RUN:-0}

# methods separated by ';', parameters selected on validation in assignment 1 for K = 256
ALL_METHODS="-exact;-lsh -k 4 -L 7 -w 0.1;-hypercube -kproj 10 -M 2000 -w 0.05 -probes 40;\
-ivfflat -kclusters 6 -nprobe 2;-ivfpq -kclusters 6 -nprobe 2 -M 16 -nbits 8"
# a pair (Q, I) verifies the same for every method, so rho, tau and Tmax are tuned on exact
VALIDATION_METHODS=${VALIDATION_METHODS-"-exact"}

# validation: base configuration, then one parameter changed at a time
VALIDATION_D=${VALIDATION_D:-10000}
BASE_C=${BASE_C:-50}
BASE_RHO=${BASE_RHO:-0.8}
BASE_TAU=${BASE_TAU:-3}
BASE_T=${BASE_T:-2000}
C_VALUES=${C_VALUES-"20 50 100"}
RHO_VALUES=${RHO_VALUES-"0.7 0.75 0.8 0.85"}
TAU_VALUES=${TAU_VALUES-"1 2 3 5"}
T_VALUES=${T_VALUES-"500 1000 2000 5000"}
# the same Tmax values with -adaptive 0, every run makes all Tmax trials
FIXED_T_VALUES=${FIXED_T_VALUES-"500 1000 2000 5000"}

# validation-grid and test: the configuration frozen after validation
# for every method, D and C
GRID_METHODS=${GRID_METHODS-$ALL_METHODS}
GRID_D_VALUES=${GRID_D_VALUES-"1000 5000 10000 25000"}
GRID_C_VALUES=${GRID_C_VALUES-"20 50 100"}
# frozen after validation: rho=0.7, tau=3, T=2000, adaptive=1
FROZEN_RHO=${FROZEN_RHO:-0.7}
FROZEN_TAU=${FROZEN_TAU:-3}
FROZEN_T=${FROZEN_T:-2000}
FROZEN_ADAPTIVE=${FROZEN_ADAPTIVE:-1}

# p = 0.99 in every run
P=${P:-0.99}
# -adaptive of every validation run except the FIXED_T_VALUES runs
ADAPTIVE=${ADAPTIVE:-1}

if [[ $DRY_RUN != 1 ]]; then
    if [[ ! -x ./verify ]]; then
        echo "error: ./verify not found, build it with make" >&2
        exit 1
    fi
    mkdir -p "$OUT_DIR"
fi

# run <method options> <set> <D> <C> <rho> <tau> <T> <adaptive>
# skips a run whose output is complete, so an interrupted study can be resumed
run() {
    local method=$1 set=$2 d=$3 c=$4 rho=$5 tau=$6 trials=$7 adaptive=$8
    local method_args
    read -r -a method_args <<< "$method"
    # e.g. validation_exact_D10000_C50_rho0.8_tau3_T2000_adaptive1
    local name="${set}_${method_args[0]#-}_D${d}_C${c}_rho${rho}_tau${tau}_T${trials}_adaptive${adaptive}"
    local output="$OUT_DIR/$name.txt"
    local log="$OUT_DIR/$name.log"
    local command=(./verify -hp "$HP" -mir "$MIR" -split "$SPLIT" -vocab "$VOCAB" -S "$S"
                   -D "$d" -set "$set" "${method_args[@]}"
                   -C "$c" -rho "$rho" -tau "$tau" -T "$trials" -p "$P" -adaptive "$adaptive"
                   -seed "$SEED" -o "$output")
    # same results with or without it, only the preprocessing is shorter
    if [[ -n $CACHE_DIR ]]; then
        command+=(-cache "$CACHE_DIR")
    fi

    if [[ $DRY_RUN == 1 ]]; then
        printf '%q ' "${command[@]}"
        echo
        return
    fi
    # verify writes the file only at the end, the summary marks a finished run
    if [[ -s $output ]] && grep -q '^MeanAfterMAP@10:' "$output"; then
        echo "skipped $name, output exists"
        return
    fi

    echo "running $name"
    local start=$SECONDS status=0
    {
        printf '%q ' "${command[@]}"
        echo
    } > "$log"
    "${command[@]}" >> "$log" 2>&1 || status=$?
    echo "wall time: $((SECONDS - start)) s" >> "$log"
    if (( status != 0 )); then
        echo "error: $name failed with status $status, see $log" >&2
        exit "$status"
    fi
}

IFS=';' read -r -a validation_methods <<< "$VALIDATION_METHODS"
IFS=';' read -r -a grid_methods <<< "$GRID_METHODS"

if [[ $phase == validation ]]; then
    # "C rho tau T adaptive" of every run, the base configuration only once
    configs=()
    add_config() {
        local config="$*" existing
        for existing in "${configs[@]}"; do
            [[ $existing == "$config" ]] && return
        done
        configs+=("$config")
    }
    add_config "$BASE_C" "$BASE_RHO" "$BASE_TAU" "$BASE_T" "$ADAPTIVE"
    for c in $C_VALUES; do add_config "$c" "$BASE_RHO" "$BASE_TAU" "$BASE_T" "$ADAPTIVE"; done
    for rho in $RHO_VALUES; do add_config "$BASE_C" "$rho" "$BASE_TAU" "$BASE_T" "$ADAPTIVE"; done
    for tau in $TAU_VALUES; do add_config "$BASE_C" "$BASE_RHO" "$tau" "$BASE_T" "$ADAPTIVE"; done
    for trials in $T_VALUES; do add_config "$BASE_C" "$BASE_RHO" "$BASE_TAU" "$trials" "$ADAPTIVE"; done
    # Tmax without the adaptive limit, so its effect on time and quality shows
    for trials in $FIXED_T_VALUES; do add_config "$BASE_C" "$BASE_RHO" "$BASE_TAU" "$trials" 0; done

    for method in "${validation_methods[@]}"; do
        for config in "${configs[@]}"; do
            read -r c rho tau trials adaptive <<< "$config"
            run "$method" validation "$VALIDATION_D" "$c" "$rho" "$tau" "$trials" "$adaptive"
        done
    done
else
    # validation-grid -> validation, test -> test
    grid_set=${phase%-grid}
    for method in "${grid_methods[@]}"; do
        for d in $GRID_D_VALUES; do
            for c in $GRID_C_VALUES; do
                run "$method" "$grid_set" "$d" "$c" \
                    "$FROZEN_RHO" "$FROZEN_TAU" "$FROZEN_T" "$FROZEN_ADAPTIVE"
            done
        done
    done
fi
