# Sandbox for the baseline variant: the coursework hand-out baked in. The
# username must match `constants.USER`, the paths `constants.ASSIGNMENT_PATH`
# and below. Rebuild after changing anything under `assignment/`, including
# dropping in the example report.
FROM python:3.12-slim
# the container has no network, so everything the agent might want is
# preinstalled: the coursework's pinned libraries (no scikit-learn, as the
# spec forbids it) and two ways to read the PDFs
COPY assignment/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt pypdf
RUN apt-get update && apt-get install -y --no-install-recommends poppler-utils && rm -rf /var/lib/apt/lists/*

ARG USERNAME=agent
RUN useradd -m -s /bin/bash $USERNAME
COPY --chown=$USERNAME:$USERNAME assignment/ /workspace/assignment/
RUN chown -R $USERNAME:$USERNAME /workspace
USER $USERNAME
WORKDIR /workspace
