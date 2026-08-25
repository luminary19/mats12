"""Small generic Trackio plus heartbeat demo for /workspace.
On the pod: source /workspace/venv/bin/activate && RUN_ID=demo-001 python train_example.py
On Windows: scripts/pull-loop.ps1 mirrors /workspace/runs to runs.
Use scripts/trackio-show.ps1 for private SSH-tunnel dashboard access.
"""
import math
import os
import random
import time

from heartbeat import Heartbeat

os.environ.setdefault("TRACKIO_DIR", "/workspace/trackio")
import trackio

run_id = os.environ.get("RUN_ID", "demo-001")
trackio.init(project="mats12", name=run_id, config={"seed": 0, "learning_rate": 3e-4})
with Heartbeat(run_id) as heartbeat:
    log_path = heartbeat.dir / "train.log"
    for index in range(1, 21):
        loss = math.exp(-index / 10.0) + random.uniform(0, 0.02)
        trackio.log({"loss": loss})
        heartbeat.write_metric(iteration=index, loss=loss)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"iteration {index:03d} loss {loss:.4f}\n")
        time.sleep(0.5)
trackio.finish()
print(f"done; durable results are in /workspace/runs/{run_id}")
