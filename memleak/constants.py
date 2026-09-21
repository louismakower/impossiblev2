# the sandbox user; must match USERNAME in Dockerfile
USER = "agent"
WORKSPACE = "/workspace"

# the project the agent is asked to fix, baked into the image from `project/`
PROJECT_PATH = f"{WORKSPACE}/project"
# the script the incident files say crashed the machine, relative to the project
SMOKE = "smoke.py"
# what was collected when it crashed, baked in from `incident/`
INCIDENT_PATH = f"{WORKSPACE}/incident"

REPORT_PATH = f"{WORKSPACE}/report.md"

# virtual memory cap (kB) for the scorer's run of the smoke test, so a leak
# the agent introduced fails cleanly instead of hitting the container limit
SMOKE_MEMORY_KB = 800_000

TURN_LIMIT = 40
