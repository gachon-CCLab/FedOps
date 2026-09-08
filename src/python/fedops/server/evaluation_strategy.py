"""Preserve strategy behavior while reporting explicit distributed evaluation."""
from flwr.server.strategy import Strategy

from .evaluation import aggregate_client_evaluations, valid_client_results, _finite


class ReportingStrategy(Strategy):
    def __init__(self, strategy, report):
        self.strategy = strategy
        self.report = report

    def initialize_parameters(self, client_manager):
        return self.strategy.initialize_parameters(client_manager)

    def configure_fit(self, server_round, parameters, client_manager):
        return self.strategy.configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(self, server_round, results, failures):
        return self.strategy.aggregate_fit(server_round, results, failures)

    def evaluate(self, server_round, parameters):
        # The callback saves the aggregated model even without server evaluation.
        return self.strategy.evaluate(server_round, parameters)

    def configure_evaluate(self, server_round, parameters, client_manager):
        instructions = self.strategy.configure_evaluate(server_round, parameters, client_manager)
        if not instructions:
            self.report(server_round, aggregate_client_evaluations([]))
        return instructions

    def aggregate_evaluate(self, server_round, results, failures):
        accepted, rejected = valid_client_results(results)
        failed = len(failures) + rejected
        if failed and not getattr(self.strategy, "accept_failures", True):
            accepted = []
        custom = {}
        if accepted:
            _, custom = self.strategy.aggregate_evaluate(server_round, accepted, failures)
        summary = aggregate_client_evaluations(accepted, failed)
        # Preserve explicitly configured strategy aggregators (e.g. F1), never
        # silently apply our sample-weighted rule to arbitrary metric names.
        extra = {key: float(value) for key, value in (custom or {}).items()
                 if key != "accuracy" and _finite(value)}
        if extra:
            summary["additional_metrics"] = extra
        self.report(server_round, summary)
        metrics = dict(extra)
        if summary["gl_accuracy"] is not None:
            metrics["accuracy"] = summary["gl_accuracy"]
        return summary["gl_loss"], metrics
