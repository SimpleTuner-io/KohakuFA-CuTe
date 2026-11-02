#!/usr/bin/env bash
# Standalone bounded allocator comparison. No installation or framework import.
set -eu
bundle=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
output=${1:?Supply a fresh external output directory}
mode=${2:-baseline}
case "$mode" in baseline|controls) ;; *) exit 2 ;; esac
case "$output" in /*) ;; *) output="$PWD/$output" ;; esac
mkdir "$output"
cd "$bundle"
sha256sum --check SHA256SUMS > "$output/input-hashes.log" 2>&1
: >&9
flock 9
old_cs=${OLD_CS:-$(command -v compute-sanitizer)}
new_cs="$bundle/sanitizer-new/compute-sanitizer"
export LD_LIBRARY_PATH="$bundle/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# Native CUBIN only; a failed load must not be repaired through PTX JIT.
export CUDA_DISABLE_PTX_JIT=1
"$old_cs" --version > "$output/old-version.txt" 2>&1
"$new_cs" --version > "$output/new-version.txt" 2>&1
if cmp -s "$output/old-version.txt" "$output/new-version.txt"; then
  printf '%s\n' identical_checker_versions > "$output/final-status.txt"
  exit 12
fi
sha256sum "$old_cs" "$new_cs" "$bundle/tmem_init" > "$output/executable-hashes.txt"
start=$EPOCHSECONDS
active=''
cleanup() {
  if test -n "$active"; then
    kill -TERM -- "-$active" 2>/dev/null || true
    sleep 1
    kill -KILL -- "-$active" 2>/dev/null || true
    wait "$active" 2>/dev/null || true
  fi
}
trap 'cleanup; exit 143' TERM
trap 'cleanup; exit 130' INT
trap cleanup EXIT
run_one() {
  local label=$1 limit=$2
  shift 2
  local remaining=$((105 - (EPOCHSECONDS - start)))
  if (( remaining <= 2 )); then
    if test "$mode" = controls; then
      mkdir "$output/$label"
      printf '%s\n' not_executed_budget_exhausted > "$output/$label.log"
      printf '%s\n' 124 > "$output/$label.status"
      printf '%s\t124\t%s\n' "$label" "$((EPOCHSECONDS-start))" >> "$output/status.tsv"
    fi
    return 124
  fi
  if (( limit > remaining )); then limit=$remaining; fi
  mkdir "$output/$label"
  setsid timeout --signal=TERM --kill-after=2s "${limit}s" "$@" "$output/$label" \
    > "$output/$label.log" 2>&1 &
  active=$!
  local status=0
  wait "$active" || status=$?
  active=''
  printf '%s\n' "$status" > "$output/$label.status"
  printf '%s\t%s\t%s\n' "$label" "$status" "$((EPOCHSECONDS-start))" >> "$output/status.tsv"
  return "$status"
}
# Controls mode changes only runtime arguments; no source or executable rebuild.
initializations=(untouched)
barriers=(roles)
if test "$mode" = controls; then
  initializations+=(sentinel)
  barriers+=(cta)
fi
for producer in allocation ordinary; do
  for initialization in "${initializations[@]}"; do
    for barrier in "${barriers[@]}"; do
      label="physical-$producer"
      if test "$mode" = controls; then label="$label-$initialization-$barrier"; fi
      if ! run_one "$label" 20 "$bundle/tmem_init" "$producer" "$initialization" "$barrier" enabled; then
        printf '%s\n' physical_control_failed > "$output/final-status.txt"
        exit 11
      fi
      grep -q '"all_match":true' "$output/$label/report.json"
      grep -q '"captured_modules":[1-9]' "$output/$label/report.json"
      grep -q "\"symbol\":\"tmem_init_$producer\"" "$output/$label/launches.jsonl"
      if test "$mode" = controls; then
        cmp "$bundle/source.cubin" "$output/$label/loaded-0.cubin"
      else
        test -s "$output/$label/loaded-0.cubin"
        test "$(od -An -tx1 -N4 "$output/$label/loaded-0.cubin" | tr -d ' \n')" = 7f454c46
        grep -a -q "tmem_init_$producer" "$output/$label/loaded-0.cubin"
      fi
    done
  done
done
# CUPTI is disabled inside initcheck. Physical capture does not expose patched ELF.
for tool in old new; do
  if test "$tool" = old; then checker=$old_cs; else checker=$new_cs; fi
  for producer in allocation ordinary; do
    for initialization in "${initializations[@]}"; do
      for barrier in "${barriers[@]}"; do
        # Old-tool baseline pair is retained without repeating its factorial.
        if test "$tool" = old && { test "$initialization" != untouched || test "$barrier" != roles; }; then continue; fi
        label="$tool-$producer"
        print_limit=6
        if test "$mode" = controls; then
          label="$label-$initialization-$barrier"
          print_limit=0
        fi
        run_one "$label" 15 "$checker" --tool initcheck --initcheck-address-space all \
          --report-api-errors no --print-limit "$print_limit" --error-exitcode 99 \
          "$bundle/tmem_init" "$producer" "$initialization" "$barrier" disabled || true
      done
    done
  done
done
printf '%s\n' 'collection_finished_not_attention_qualification' > "$output/final-status.txt"
printf '%s\n' "$((EPOCHSECONDS-start))" > "$output/elapsed-seconds.txt"
