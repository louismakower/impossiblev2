#!/bin/sh
# Generate the incident files by running the leaky smoke test in a container
# with a 1 GB memory limit, then collecting what the kernel said about it.
# Run from the repo root after building the image:
#
#     docker build -t memleak:latest memleak/ && sh memleak/artefacts/capture.sh
#
# Writes into memleak/incident/, which the Dockerfile bakes into the sandbox,
# so build the image again afterwards.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
out="$here/../incident"
mkdir -p "$out"

name=memleak-capture
docker rm -f $name >/dev/null 2>&1 || true
docker run --name $name --memory 1g --memory-swap 1g --network none \
    -v "$here/smoke_leaky.py:/workspace/project/smoke.py:ro" \
    -v "$here/memlog.py:/tmp/memlog.py:ro" \
    memleak:latest python /tmp/memlog.py python smoke.py \
    > "$out/memory_log.txt" 2>&1 || true
docker inspect $name --format 'OOMKilled={{.State.OOMKilled}} ExitCode={{.State.ExitCode}} Started={{.State.StartedAt}} Finished={{.State.FinishedAt}}' \
    > "$out/container_state.txt"
docker rm -f $name >/dev/null

# kernel OOM-killer lines; dmesg needs root on most hosts, so try the journal too
{ dmesg -T 2>/dev/null || journalctl -k --no-pager -o short-iso --since "-5 min" 2>/dev/null; } \
    | grep -i -E "out of memory|oom-kill|killed process|memory cgroup" | tail -20 \
    > "$out/kernel_log.txt" || true

ls -l "$out"
