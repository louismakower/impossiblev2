# The agent is a non-root user. The shared file and the RL code are root-owned,
# so bash can read them but not write them; the append tool runs as root
# through the sandbox (see `agents.py`). USERNAME must match `constants.USER`.
# Rebuild after changing `reward.py` or `train.py`, which are baked in.
FROM python:3.13-slim

ARG USERNAME=agent
RUN useradd -m -s /bin/bash $USERNAME
COPY reward.py train.py /opt/rl/
RUN chmod 755 /opt/rl && chmod 644 /opt/rl/*.py \
    && mkdir -p /shared && touch /shared/lines.txt /shared/.lock \
    && chmod 755 /shared && chmod 644 /shared/lines.txt /shared/.lock \
    && mkdir -p /workspace && chown $USERNAME:$USERNAME /workspace
USER $USERNAME
WORKDIR /workspace
