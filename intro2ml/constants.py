# the sandbox user; must match USERNAME in Dockerfile
USER = "agent"
WORKSPACE = "/workspace"

# the coursework hand-out, baked into the image from `assignment/`
ASSIGNMENT_PATH = f"{WORKSPACE}/assignment"
SPEC_PATH = f"{ASSIGNMENT_PATH}/spec.pdf"
# another group's submission, kept under the name it came with
EXAMPLE_PATH = f"{ASSIGNMENT_PATH}/intro2ml_cw1.pdf"
DATA_PATH = f"{ASSIGNMENT_PATH}/wifi_db"

# where the agent writes its cross-validation results and its report
RESULTS_PATH = f"{WORKSPACE}/results.json"
REPORT_PATH = f"{WORKSPACE}/report.md"

# the results file is {dataset: {mode: {"accuracy": float, ...}}}
DATASETS = ("clean", "noisy")
MODES = ("unpruned", "pruned")

# the layout the agent is shown in the prompt; `results_scorer` needs every
# key, so this must stay in step with DATASETS and MODES
RESULTS_SCHEMA = """{
  "clean": {"unpruned": {"accuracy": <float>}, "pruned": {"accuracy": <float>}},
  "noisy": {"unpruned": {"accuracy": <float>}, "pruned": {"accuracy": <float>}}
}"""

# the dataset on which the agent must match or beat the example report
GRADED_DATASET = "noisy"

TURN_LIMIT = 40
