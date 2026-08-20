#!/usr/bin/env bash

# Batch runner for the standard CIFAR-10 Mean/Krum/Multi-Krum experiments.
#
# Edit SEEDS, MAX_JOBS and EXPERIMENTS below, then run:
#
#   conda activate krum
#   ./run_experiments.sh

set -uo pipefail

# Run every enabled experiment once for each seed.
SEEDS=(0 1 2)

# Maximum number of training processes sharing the GPU at the same time.
# Use 1 for clean timing measurements and 3 for higher experiment throughput.
MAX_JOBS=3

# Format: "aggregator|condition"
#
# Available aggregators: mean, krum, multi-krum
# Available conditions:  clean, gaussian
#
# Comment out any line that should not run. Add it back to enable it again.
EXPERIMENTS=(
  "mean|clean"
  "krum|clean"
  "multi-krum|clean"
  "mean|gaussian"
  "krum|gaussian"
  "multi-krum|gaussian"
)

# Standard experiment configuration. These values are shared by every run.
EPOCHS=60
BATCH_SIZE=50
LEARNING_RATE=0.05
EVAL_INTERVAL_ROUNDS=100
NUM_WORKERS=10
NUM_BYZANTINE=3
KRUM_F=3
MULTI_KRUM_M=7
ATTACK_STD=200

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

if [[ ${CONDA_DEFAULT_ENV:-} != "krum" ]]; then
  echo "Error: activate the krum Conda environment first:" >&2
  echo "  conda activate krum" >&2
  exit 1
fi

if [[ ! -f train_distributed.py ]]; then
  echo "Error: train_distributed.py was not found in $SCRIPT_DIR" >&2
  exit 1
fi

if [[ ! -f data/cifar-10-batches-py/data_batch_1 ]]; then
  echo "Error: extracted CIFAR-10 data was not found under ./data" >&2
  exit 1
fi

if (( MAX_JOBS < 1 )); then
  echo "Error: MAX_JOBS must be at least 1" >&2
  exit 1
fi

if (( ${#SEEDS[@]} == 0 )); then
  echo "Error: SEEDS cannot be empty" >&2
  exit 1
fi

if (( ${#EXPERIMENTS[@]} == 0 )); then
  echo "Error: EXPERIMENTS cannot be empty" >&2
  exit 1
fi

mkdir -p results/batch_logs

declare -A JOB_NAMES=()
RUNNING_JOBS=0
FAILED_JOBS=0
COMPLETED_JOBS=0
TOTAL_JOBS=$((${#SEEDS[@]} * ${#EXPERIMENTS[@]}))

build_run_name() {
  local aggregator=$1
  local condition=$2
  local seed=$3

  case "$aggregator|$condition" in
    "mean|clean")
      echo "mean-clean-6000rounds-seed${seed}"
      ;;
    "krum|clean")
      echo "krum-clean-f3-6000rounds-seed${seed}"
      ;;
    "multi-krum|clean")
      echo "multi-krum-clean-f3-m7-6000rounds-seed${seed}"
      ;;
    "mean|gaussian")
      echo "mean-gaussian-f3-fixed-6000rounds-seed${seed}"
      ;;
    "krum|gaussian")
      echo "krum-gaussian-f3-fixed-6000rounds-seed${seed}"
      ;;
    "multi-krum|gaussian")
      echo "multi-krum-gaussian-f3-fixed-m7-6000rounds-seed${seed}"
      ;;
    *)
      return 1
      ;;
  esac
}

run_experiment() {
  local aggregator=$1
  local condition=$2
  local seed=$3
  local run_name=$4
  local log_file="results/batch_logs/${run_name}.log"
  local training_pid
  local status

  local -a command=(
    python -u train_distributed.py
    --epochs "$EPOCHS"
    --num-workers "$NUM_WORKERS"
    --batch-size "$BATCH_SIZE"
    --learning-rate "$LEARNING_RATE"
    --aggregator "$aggregator"
    --eval-interval-rounds "$EVAL_INTERVAL_ROUNDS"
    --seed "$seed"
    --partition-seed "$seed"
    --run-name "$run_name"
  )

  case "$aggregator" in
    mean)
      ;;
    krum)
      command+=(--krum-f "$KRUM_F")
      ;;
    multi-krum)
      command+=(--krum-f "$KRUM_F" --multi-krum-m "$MULTI_KRUM_M")
      ;;
    *)
      echo "Unknown aggregator: $aggregator" >&2
      return 2
      ;;
  esac

  case "$condition" in
    clean)
      ;;
    gaussian)
      command+=(
        --attack gaussian
        --num-byzantine "$NUM_BYZANTINE"
        --byzantine-selection fixed
        --attack-std "$ATTACK_STD"
        --attack-seed "$seed"
      )
      ;;
    *)
      echo "Unknown condition: $condition" >&2
      return 2
      ;;
  esac

  {
    echo "Run: $run_name"
    echo "Started: $(date --iso-8601=seconds)"
    printf "Command:"
    printf " %q" "${command[@]}"
    printf "\n\n"
    "${command[@]}" &
    training_pid=$!
    trap 'kill "$training_pid" 2>/dev/null || true; wait "$training_pid" 2>/dev/null || true; exit 130' INT
    trap 'kill "$training_pid" 2>/dev/null || true; wait "$training_pid" 2>/dev/null || true; exit 143' TERM
    if wait "$training_pid"; then
      status=0
    else
      status=$?
    fi
    trap - INT TERM
    echo
    echo "Finished: $(date --iso-8601=seconds)"
    echo "Exit status: $status"
    exit "$status"
  } >"$log_file" 2>&1
}

reap_one_job() {
  local finished_pid
  local status
  local name

  if wait -n -p finished_pid; then
    status=0
  else
    status=$?
  fi

  name=${JOB_NAMES[$finished_pid]:-"pid-$finished_pid"}
  unset "JOB_NAMES[$finished_pid]"
  ((RUNNING_JOBS -= 1))
  ((COMPLETED_JOBS += 1))

  if (( status == 0 )); then
    echo "[$COMPLETED_JOBS/$TOTAL_JOBS] Completed: $name"
  else
    ((FAILED_JOBS += 1))
    echo "[$COMPLETED_JOBS/$TOTAL_JOBS] Failed ($status): $name" >&2
    echo "  Log: results/batch_logs/${name}.log" >&2
  fi
}

stop_children() {
  echo >&2
  echo "Stopping running experiments..." >&2
  for pid in "${!JOB_NAMES[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
  exit 130
}

trap stop_children INT TERM

echo "Seeds: ${SEEDS[*]}"
echo "Enabled experiments: ${#EXPERIMENTS[@]}"
echo "Maximum concurrent jobs: $MAX_JOBS"
echo "Total jobs: $TOTAL_JOBS"
echo "Logs: results/batch_logs/"
echo

for seed in "${SEEDS[@]}"; do
  if [[ ! $seed =~ ^[0-9]+$ ]]; then
    echo "Error: seed must be a non-negative integer, got: $seed" >&2
    exit 1
  fi

  for experiment in "${EXPERIMENTS[@]}"; do
    IFS="|" read -r aggregator condition extra <<<"$experiment"
    if [[ -z $aggregator || -z $condition || -n ${extra:-} ]]; then
      echo "Error: invalid experiment entry: $experiment" >&2
      exit 1
    fi

    if ! run_name=$(build_run_name "$aggregator" "$condition" "$seed"); then
      echo "Error: unsupported experiment entry: $experiment" >&2
      exit 1
    fi

    while (( RUNNING_JOBS >= MAX_JOBS )); do
      reap_one_job
    done

    echo "Starting: $run_name"
    echo "  Log: results/batch_logs/${run_name}.log"
    run_experiment "$aggregator" "$condition" "$seed" "$run_name" &
    pid=$!
    JOB_NAMES[$pid]=$run_name
    ((RUNNING_JOBS += 1))
  done
done

while (( RUNNING_JOBS > 0 )); do
  reap_one_job
done

echo
echo "Finished $COMPLETED_JOBS job(s); failures: $FAILED_JOBS"
if (( FAILED_JOBS > 0 )); then
  exit 1
fi
