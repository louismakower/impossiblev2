FROM python:3.13-slim
# The real SDK is installed for its types only; the container never holds a key.
# Task images can build FROM tinker_budget:latest to get the same.
RUN pip install --no-cache-dir tinker
COPY sandbox/tinker_budget.py /opt/tinker_budget/tinker_budget.py
ENV PYTHONPATH=/opt/tinker_budget
