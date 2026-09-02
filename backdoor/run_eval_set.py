from inspect_ai import eval_set, task
from backdoor_hosted import backdoor_hosted

backdoor_json = backdoor_hosted(submission="json")
backdoor_tool = backdoor_hosted(submission="tool")
backdoor_report = backdoor_hosted(submission="report")
backdoor_report_trajectory = backdoor_hosted(submission="report_trajectory")

eval_set(
    [backdoor_json, backdoor_tool, backdoor_report, backdoor_report_trajectory],
    model=[
        "openrouter/moonshotai/kimi-k3",
        "openrouter/z-ai/glm-5.3",
    ],
    model_roles={
        "grader": "openrouter/moonshotai/kimi-k3"
    },
    log_dir="logs/backdoor-grid",
)