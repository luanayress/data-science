"""Optional shadow inference that never changes the champion response."""

import logging
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class ShadowPredictionService:
    def __init__(self, champion, challenger):
        self.champion = champion
        self.challenger = challenger

    def compare(self, frame, champion_result: Dict, request_id: Optional[str] = None) -> Dict:
        try:
            challenger_result = self.challenger.predict_with_confidence(frame)
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning(
                "Shadow prediction skipped request_id=%s challenger_version=%s reason=%s",
                request_id, self.challenger.version, exc,
            )
            return {"status": "skipped", "reason": str(exc)}
        champion_probabilities = np.asarray(champion_result["probabilities"])
        challenger_probabilities = np.asarray(challenger_result["probabilities"])
        deltas = challenger_probabilities - champion_probabilities
        agreements = np.asarray(champion_result["predictions"]) == np.asarray(challenger_result["predictions"])
        for index in range(len(frame)):
            logger.info(
                "Shadow prediction request_id=%s champion_version=%s challenger_version=%s champion_probability=%.8f challenger_probability=%.8f probability_delta=%.8f champion_prediction=%d challenger_prediction=%d",
                request_id, self.champion.version, self.challenger.version,
                champion_probabilities[index], challenger_probabilities[index], deltas[index],
                champion_result["predictions"][index], challenger_result["predictions"][index],
            )
        return {
            "agreement_rate": float(np.mean(agreements)),
            "mean_probability_delta": float(np.mean(deltas)),
            "median_probability_delta": float(np.median(deltas)),
            "p95_probability_delta": float(np.percentile(np.abs(deltas), 95)),
        }
