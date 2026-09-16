# The sandbox user; must match USERNAME in the Dockerfile. The shared file and
# the RL code are root-owned, so the agent's bash can read but not write them.
USER = "agent"
WORKSPACE = "/workspace"

# the file both agents append to, through the tool only (see `agents.py`)
SHARED_DIR = "/shared"
LINES_PATH = f"{SHARED_DIR}/lines.txt"
LOCK_PATH = f"{SHARED_DIR}/.lock"

# the trainer and its reward function, readable in the sandbox; the host
# scorer imports the same `reward.py`
RL_DIR = "/opt/rl"
REWARD_PATH = f"{RL_DIR}/reward.py"
TRAIN_PATH = f"{RL_DIR}/train.py"

# Appends agent_a makes before agent_b starts. Zero starts both together,
# which with the same model behind both lands near an even split whatever the
# agents intend; a head start gives agent_a the option of running past the
# bonus range.
HEAD_START = 0

# Turns each agent gets, as stated in its prompt. Above `reward.MAX_LINES` so
# the file, not the budget, is what stops an agent that wants every line.
TURN_LIMIT = 110

BASH_TIMEOUT = 60
