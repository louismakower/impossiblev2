# The agent is a non-root user. The shared file and the RL code are root-owned,
# so bash can read them but not write them; the append tool runs as root
# through the sandbox (see `agents.py`). USERNAME must match `constants.USER`.
# `reward.py` and `train.py` are installed into /opt/rl at sample start
# (`install.py`), so changing them needs no rebuild.
FROM python:3.13-slim

ARG USERNAME=agent
RUN useradd -m -s /bin/bash $USERNAME
RUN mkdir -p /opt/rl && chmod 755 /opt/rl \
    && mkdir -p /shared && touch /shared/lines.txt /shared/.lock \
    && chmod 755 /shared && chmod 644 /shared/lines.txt /shared/.lock \
    && mkdir -p /workspace && chown $USERNAME:$USERNAME /workspace
USER $USERNAME
WORKDIR /workspace
