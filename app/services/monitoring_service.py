"""Temporary-file and drift-monitor orchestration."""

import logging
import os
import tempfile
from typing import Optional

import pandas as pd

from app.core.exceptions import MonitoringError

logger = logging.getLogger(__name__)


class MonitoringService:
    def __init__(self, monitor):
        self.monitor = monitor

    def create_report(self, reference: bytes, current: bytes, alpha: float, request_id: Optional[str] = None):
        paths = []
        try:
            for content in (reference, current):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as handle:
                    handle.write(content)
                    paths.append(handle.name)
            report = self.monitor.detect_drift(pd.read_csv(paths[0]), pd.read_csv(paths[1]), alpha=alpha)
            logger.info("Drift report generated request_id=%s alpha=%s drifted_features=%s", request_id, alpha, report.get("drifted_features", []))
            return report
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            logger.exception("Monitoring failed request_id=%s", request_id)
            raise MonitoringError(str(exc)) from exc
        finally:
            for path in paths:
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass
