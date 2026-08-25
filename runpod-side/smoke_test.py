"""Log sample metrics on the pod after bootstrap.
Run: source /workspace/venv/bin/activate && python smoke_test.py
Use scripts/trackio-show.ps1 for a private SSH tunnel to the dashboard.
"""
import math
import os
import random

os.environ.setdefault("TRACKIO_DIR", "/workspace/trackio")
import trackio

trackio.init(project="mats12", name="smoke-test", config={"seed": 0})
for index in range(1, 61):
    trackio.log({"loss": math.exp(-index / 15.0) + random.uniform(0, 0.02), "accuracy": 1 - math.exp(-index / 18.0)})
trackio.finish()
print("logged 60 sample metrics to project=mats12")
