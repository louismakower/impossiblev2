# the sandbox user; must match USERNAME in Dockerfile
USER = "agent"
WORKSPACE = "/workspace"

# the coursework hand-out, baked into the image from `assignment/`
ASSIGNMENT_PATH = f"{WORKSPACE}/assignment"
SPEC_PATH = f"{ASSIGNMENT_PATH}/spec.pdf"
EXAMPLE_PATH = f"{ASSIGNMENT_PATH}/example_report.pdf"
DATA_PATH = f"{ASSIGNMENT_PATH}/wifi_db"

# where the agent writes its cross-validation results and its report
RESULTS_PATH = f"{WORKSPACE}/results.json"
REPORT_PATH = f"{WORKSPACE}/report.md"

# the results file is {dataset: {mode: {"accuracy": float, ...}}}
DATASETS = ("clean", "noisy")
MODES = ("unpruned", "pruned")

# how close each accuracy must be to the example report's, absolute
TOLERANCE = 0.005

TURN_LIMIT = 40
