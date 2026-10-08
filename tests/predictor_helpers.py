"""Small in-memory predictor used by event-driven tests."""

from edge_moe_sim.schema import PredictorOutput, RoutingEvent


class FixedPredictor:
    """Return fixed candidates and a fixed ready offset."""

    def __init__(self, ids, scores=None, ready=0) -> None:
        self.ids = tuple(ids)
        self.scores = tuple(scores or [1.0 - 0.1 * index for index in range(len(ids))])
        self.ready = ready

    def predict(self, event: RoutingEvent, layer_start_cycle: int) -> PredictorOutput:
        """Build an output keyed to the supplied routing event."""
        return PredictorOutput(
            event.request_id, event.token_position, event.layer_id,
            self.ids, self.scores, self.ready,
        )

