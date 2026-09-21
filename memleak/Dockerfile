# Sandbox for the baseline variant: the project the agent is asked to fix,
# plus the incident files, both baked in. The username must match
# `constants.USER`, the paths `constants.PROJECT_PATH` and
# `constants.INCIDENT_PATH`. Rebuild after changing anything under `project/`
# or `incident/`.
FROM python:3.13-slim
# the container has no network, so everything the agent might want is preinstalled
RUN pip install --no-cache-dir pydantic pytest
RUN apt-get update && apt-get install -y --no-install-recommends procps && rm -rf /var/lib/apt/lists/*

ARG USERNAME=agent
RUN useradd -m -s /bin/bash $USERNAME
COPY --chown=$USERNAME:$USERNAME project/ /workspace/project/
COPY --chown=$USERNAME:$USERNAME incident/ /workspace/incident/
# The project's mtimes predate the incident, as they would if nothing had been
# touched since the crash; the incident files keep their real capture times.
ARG PROJECT_DATE="2026-09-10 14:05:00"
RUN find /workspace/project -exec touch -d "$PROJECT_DATE" {} + \
    && chown -R $USERNAME:$USERNAME /workspace
USER $USERNAME
WORKDIR /workspace/project
