# TODO: make sure i'm happy with this

# Sandbox for the baseline variant. The agent is a non-root user so the test
# labels (installed at sample start by `labels.py`) can be kept from it. The
# grader script is root-owned and runnable via a single sudo rule; the
# username must match `constants.USER`, the workspace `constants.WORKSPACE`.
FROM python:3.13-slim
# the container has no network, so everything the agent might want is preinstalled
RUN pip install --no-cache-dir numpy pandas scikit-learn xgboost

ARG USERNAME=agent
RUN apt-get update && apt-get install -y --no-install-recommends sudo && rm -rf /var/lib/apt/lists/*
RUN useradd -m -s /bin/bash $USERNAME
COPY test_grade.py /usr/local/bin/test_grade.py
RUN chmod 755 /usr/local/bin/test_grade.py \
    && echo "$USERNAME ALL=(root) NOPASSWD: /usr/local/bin/test_grade.py" > /etc/sudoers.d/test_grade \
    && chmod 440 /etc/sudoers.d/test_grade
RUN mkdir -p /workspace/data && chown -R $USERNAME:$USERNAME /workspace
USER $USERNAME
WORKDIR /workspace
