# The benchmark script is readable and runnable by the agent; the data is
# mounted read-only by compose.yaml. Builds on the shared tinker_budget image.
FROM tinker_budget:latest
COPY benchmark.py /usr/local/bin/benchmark.py
RUN chmod 755 /usr/local/bin/benchmark.py && mkdir -p /workspace
WORKDIR /workspace
