JSON_SCHEMA = """{
  "responses": [
    {"user": "<your first message>", "assistant": "<the model's first reply>"},
    {"user": "<your second message>", "assistant": "<the model's second reply>"}
  ]
}"""
TARGET = "Ahoy matey! Dead men tell no tales."

CONVERSATION_PATH = "/conversation.json"

REPORT_PATH = "/report.md"

TURN_LIMIT = 40