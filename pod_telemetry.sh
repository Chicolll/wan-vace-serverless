#!/bin/bash
# Reusable pod telemetry capture. 2026-06-21 learning: a metered benchmark MUST capture phase + the RIGHT signals to
# the VOLUME (not the ephemeral pod disk), or the spend taught nothing. GPU utilization ALONE is blind where our two
# cost-dominant phases live, so this captures the full set:
#   - GPU compute (sm%), GPU mem-bandwidth (mem%), PCIe rx/tx (the USP All2All comms signal), power, clocks, fb mem
#     -> via `nvidia-smi dmon` (one line per GPU per interval). This is what answers "comms-bound vs compute-bound".
#   - GPU memory USED in MiB + util%               -> via --query-gpu csv (dmon's mem col is bandwidth%, not MiB).
#   - CPU% + RAM used                              -> via /proc (the COLD GGUF dequant is CPU-bound w/ GPUs at 0% —
#                                                     GPU telemetry can't see it; this is the production blocker).
#   - Disk read/write bytes                        -> via /proc/diskstats (model load from the volume / tar extract).
# All to /workspace/telemetry/<run>/ (the VOLUME, survives pod stop). No extra packages — nvidia-smi + /proc only.
#
# Usage (on the pod):
#   bash pod_telemetry.sh start <run>          # begin all loggers; prints PIDs
#   bash pod_telemetry.sh phase <run> <name>   # stamp a phase boundary (extract_done/load_done/sample_start/decode_done)
#   bash pod_telemetry.sh save  <run>          # copy comfy.log + bench/install/extract logs to the volume
#   bash pod_telemetry.sh stop  <run>          # kill the loggers (do this before pod stop, after save)
# After: STOP the pod, then read /workspace/telemetry/<run>/ from any pod (or S3):
#   gpu_dmon.txt (sm/mem-bw/pcie/power/clk per GPU), gpu_mem.csv (MiB used + util%), sys.csv (CPU%+RAM),
#   disk.csv (read/write MiB), phases.log (boundaries), *.log (per-step ComfyUI timings).
set -u
TELE="${TELE_DIR:-/workspace/telemetry}"
cmd="${1:-}"; run="${2:-run}"; dir="$TELE/$run"
mkdir -p "$dir"
ts() { date +%s.%3N; }

case "$cmd" in
  start)
    # 1) per-GPU pcie/comms + sm/mem-bw/power/clk/fbmem at dmon's finest (1s) — cross-check + the PCIe columns
    #    that query-gpu lacks. (dmon minimum interval is 1s; the 5 Hz logger below resolves within-step structure.)
    setsid nice -n 19 bash -c "nvidia-smi dmon -s pucmt -d 1 -o DT > '$dir/gpu_dmon.txt' 2>&1" </dev/null >/dev/null 2>&1 &
    p1=$!
    # 2) GPU snapshot signals @ 5 Hz (-lms 200): sm%, mem-bw%, VRAM MiB, power, clocks, temp. ONE persistent
    #    process (no per-sample fork), niced. Sub-step per-layer detail = the profiler.
    setsid nice -n 19 bash -c "nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm,temperature.gpu --format=csv,nounits -lms 200 > '$dir/gpu_hifreq.csv' 2>&1" </dev/null >/dev/null 2>&1 &
    p2=$!
    # 3) CPU% (whole-box, from /proc/stat delta) + RAM used MiB, every 1s -> catches the CPU-bound cold load
    setsid nice -n 19 bash -c '
      echo "ts,cpu_pct,mem_used_mib,mem_total_mib" > "'"$dir"'/sys.csv"
      read a b c d e f g h i j < /proc/stat; prev=$((b+c+d+f+g+h+i)); previdle=$e
      while true; do
        sleep 1
        read a b c d e f g h i j < /proc/stat; tot=$((b+c+d+f+g+h+i)); idle=$e
        dt=$((tot-prev)); di=$((idle-previdle)); prev=$tot; previdle=$idle
        cpu=0; [ $((dt+di)) -gt 0 ] && cpu=$(( (100*dt)/(dt+di) ))
        mt=$(awk "/MemTotal/{print int(\$2/1024)}" /proc/meminfo)
        ma=$(awk "/MemAvailable/{print int(\$2/1024)}" /proc/meminfo)
        echo "$(date +%s.%3N),$cpu,$((mt-ma)),$mt" >> "'"$dir"'/sys.csv"
      done' </dev/null >/dev/null 2>&1 &
    p3=$!
    # 4) disk read/write MiB cumulative (all block devs summed) -> model load / tar extract throughput
    setsid nice -n 19 bash -c '
      echo "ts,read_mib,write_mib" > "'"$dir"'/disk.csv"
      while true; do
        rs=$(awk "{r+=\$6; w+=\$10} END{print r,w}" /proc/diskstats)
        set -- $rs
        echo "$(date +%s.%3N),$(( $1*512/1048576 )),$(( $2*512/1048576 ))" >> "'"$dir"'/disk.csv"
        sleep 1
      done' </dev/null >/dev/null 2>&1 &
    p4=$!
    # 5) NVLink throughput per GPU (the inter-GPU comms signal that doesn't exist on single-GPU) -> nvlink.csv
    #    Counters are CUMULATIVE → per-phase total volume is exact at ANY cadence; 2s only sets curve time-resolution
    #    (enough for the per-step comms pattern). Sub-step per-layer detail = the profiler. This is the heaviest logger
    #    (forks nvidia-smi each tick), so it's backed off the most.
    setsid nice -n 19 bash -c '
      echo "ts,raw" > "'"$dir"'/nvlink.csv"
      while true; do
        v=$(nvidia-smi nvlink -gt d 2>/dev/null | tr "\n" ";" | tr -s " ")
        echo "$(date +%s.%3N),$v" >> "'"$dir"'/nvlink.csv"
        sleep 2
      done' </dev/null >/dev/null 2>&1 &
    p5=$!
    # 6) per-CORE CPU (how many cores the N-way dequant/workers actually use) -> percpu.csv
    setsid nice -n 19 bash -c '
      echo "ts,percpu_busy_pct" > "'"$dir"'/percpu.csv"
      declare -A pt pi
      while true; do
        sleep 1; line=""
        while read c a b d e f g h rest; do
          case "$c" in cpu[0-9]*)
            tot=$((a+b+d+f+g+h)); idle=$e
            dt=$((tot-${pt[$c]:-0})); di=$((idle-${pi[$c]:-0})); pt[$c]=$tot; pi[$c]=$idle
            p=0; [ $((dt+di)) -gt 0 ] && p=$(((100*dt)/(dt+di)))
            line="$line$c:$p " ;;
          esac
        done < /proc/stat
        echo "$(date +%s.%3N),$line" >> "'"$dir"'/percpu.csv"
      done' </dev/null >/dev/null 2>&1 &
    p6=$!
    # 7) per-PROCESS (the Ray workers): top CPU/RAM procs + process->GPU memory map -> proc.csv, gpu_apps.csv
    setsid nice -n 19 bash -c '
      echo "ts,pid,pcpu,pmem,rss_mib,comm" > "'"$dir"'/proc.csv"
      echo "ts,raw" > "'"$dir"'/gpu_apps.csv"
      while true; do
        t=$(date +%s.%3N)
        ps -eo pid,pcpu,pmem,rss,comm --sort=-pcpu 2>/dev/null | awk -v t="$t" "NR>1 && NR<=9 {print t\",\"\$1\",\"\$2\",\"\$3\",\"int(\$4/1024)\",\"\$5}" >> "'"$dir"'/proc.csv"
        g=$(nvidia-smi --query-compute-apps=pid,used_memory,gpu_uuid --format=csv,noheader 2>/dev/null | tr "\n" ";")
        echo "$t,$g" >> "'"$dir"'/gpu_apps.csv"
        sleep 3
      done' </dev/null >/dev/null 2>&1 &
    p7=$!
    # 8) network I/O (cumulative RX/TX MiB across non-loopback ifaces, 1s) -> the MooseFS volume READ rate, which is
    #    the cold model-load bottleneck and is INVISIBLE to /proc/diskstats (network FS). Captured per the 2026-06-22
    #    finding that the cold GGUF load is a network read with no local-disk/CPU signal.
    setsid nice -n 19 bash -c '
      echo "ts,rx_mib,tx_mib" > "'"$dir"'/net.csv"
      while true; do
        v=$(awk "/:/ && !/lo:/{gsub(/.*:/,\"\"); r+=\$1; t+=\$9} END{print int(r/1048576), int(t/1048576)}" /proc/net/dev)
        set -- $v
        echo "$(date +%s.%3N),${1:-0},${2:-0}" >> "'"$dir"'/net.csv"
        sleep 1
      done' </dev/null >/dev/null 2>&1 &
    p8=$!
    echo "$p1 $p2 $p3 $p4 $p5 $p6 $p7 $p8" > "$dir/.tele_pids"
    echo "$(ts) telemetry_start" >> "$dir/phases.log"
    echo "TELE_START pids=$p1..$p7 -> $dir/{gpu_dmon.txt,gpu_mem.csv,sys.csv,disk.csv,nvlink.csv,percpu.csv,proc.csv,gpu_apps.csv}" ;;
  phase)
    echo "$(ts) ${3:-phase}" >> "$dir/phases.log"
    echo "PHASE_STAMPED ${3:-phase} @ $(ts)" ;;
  save)
    for f in /root/comfy.log /root/comfy_final.log /root/bench_*.log /root/install.log /root/extract.log; do
      [ -f "$f" ] && cp "$f" "$dir/" 2>/dev/null
    done
    echo "$(ts) telemetry_save" >> "$dir/phases.log"
    echo "TELE_SAVED -> $dir/ : $(ls "$dir" 2>/dev/null | tr '\n' ' ')" ;;
  stop)
    [ -f "$dir/.tele_pids" ] && kill $(cat "$dir/.tele_pids") 2>/dev/null
    echo "$(ts) telemetry_stop" >> "$dir/phases.log"
    echo "TELE_STOPPED $(cat "$dir/.tele_pids" 2>/dev/null)" ;;
  *)
    echo "usage: pod_telemetry.sh start|phase|save|stop <run> [phase_name]" ;;
esac
