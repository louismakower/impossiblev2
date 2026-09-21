"""Run a command and log its resident memory until it exits.

    python memlog.py CMD [ARG...]

One line per sample; a final line with the exit status and peak. Used by
`capture.sh` to record the leaky smoke test climbing until the container's
memory limit kills it.
"""

import datetime
import subprocess
import sys
import time


def rss_kb(pid: int) -> int | None:
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except FileNotFoundError:
        pass
    return None


def main():
    proc = subprocess.Popen(sys.argv[1:])
    peak = 0
    while proc.poll() is None:
        rss = rss_kb(proc.pid)
        if rss is not None:
            peak = max(peak, rss)
            stamp = datetime.datetime.now().isoformat(timespec="milliseconds")
            print(f"{stamp} pid={proc.pid} rss={rss / 1024:.0f}MiB", flush=True)
        time.sleep(0.25)
    print(f"exit={proc.returncode} peak={peak / 1024:.0f}MiB", flush=True)


if __name__ == "__main__":
    main()
