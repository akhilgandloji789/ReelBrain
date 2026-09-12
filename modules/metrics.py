import time
from typing import Any


class TelemetryMetrics:
    def __init__(self):
        self.start_time = time.time()
        self.total_processed = 0
        self.total_errors = 0
        self.last_latency = 0.0

    def record_run(self, duration: float, success: bool) -> None:
        self.last_latency = duration
        if success:
            self.total_processed += 1
        else:
            self.total_errors += 1

    def get_status(self) -> dict[str, Any]:
        uptime_mins = int((time.time() - self.start_time) / 60)
        return {
            "uptime_mins": uptime_mins,
            "processed": self.total_processed,
            "errors": self.total_errors,
            "last_latency_sec": round(self.last_latency, 2)
        }
