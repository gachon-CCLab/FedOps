import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from omegaconf import OmegaConf
from flwr.common import Code, EvaluateRes, Status, ndarrays_to_parameters
from flwr.server.strategy import FedAvg
from fedops.server.evaluation import (
    aggregate_client_evaluations, evaluation_policy, prepare_validation_loader,
    validate_relative_path, validation_directory,
)
from fedops.server.evaluation_strategy import ReportingStrategy


def result(cid, n, loss, accuracy=None):
    return (SimpleNamespace(cid=cid), EvaluateRes(Status(Code.OK, ''), loss, n,
            {} if accuracy is None else {"accuracy": accuracy, "f1": 0.123}))


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_disabled_never_calls_data_loader(self):
        factory = Mock(side_effect=AssertionError('must not read server data'))
        self.assertIsNone(prepare_validation_loader({"server_evaluation": {"enabled": False}}, factory))
        factory.assert_not_called()

    def test_missing_enabled_keeps_legacy(self):
        cfg = OmegaConf.create({"batch_size": 8, "dataset": {"root": "./dataset", "download": True}})
        factory = Mock(return_value=[1])
        self.assertNotIn('enabled', evaluation_policy(cfg))
        prepare_validation_loader(cfg, factory)
        factory.assert_called_once_with(batch_size=8, data_root='./dataset', download=True)

    def test_requires_explicit_real_directory(self):
        with self.assertRaisesRegex(ValueError, 'directory'):
            validation_directory({"enabled": True})
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, 'empty'):
                validation_directory({"data_root": root})
            Path(root, 'data.csv').write_text('x,y\n1,2\n')
            cfg = OmegaConf.create({"batch_size": 8, "server_evaluation": {"enabled": True, "data_root": root}})
            factory = Mock(return_value=[1])
            prepare_validation_loader(cfg, factory)
            factory.assert_called_once_with(batch_size=8, data_root=str(Path(root).resolve()), download=False)
            with self.assertRaisesRegex(ValueError, 'no evaluation'):
                prepare_validation_loader(cfg, Mock(return_value=[]))

    def test_path_rejects_traversal_absolute_and_symlink(self):
        for value in ('', '../x', '/etc', 'a/../b', 'a//b', 'a\\b', 'a\nx'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_relative_path(value)
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as outside:
            directory = Path(root, 'validation')
            directory.mkdir()
            (directory / 'escape').symlink_to(outside)
            with self.assertRaisesRegex(ValueError, 'symlink'):
                validation_directory({"data_root": str(directory), "allowed_root": root})

    def test_campaign_override_and_strict_boolean(self):
        with patch.dict(os.environ, {"FEDOPS_CAMPAIGN_CONFIG": json.dumps({"serverEvaluation": {"enabled": False}})}):
            self.assertFalse(evaluation_policy({"server_evaluation": {"enabled": True}})['enabled'])
        for value in ('false', 0, None):
            with self.assertRaises(ValueError):
                evaluation_policy({"server_evaluation": {"enabled": value}})

    def test_same_release_can_switch_campaign_modes_without_mutating_defaults(self):
        config = OmegaConf.create({"batch_size": 8, "server_evaluation": {"enabled": False}})
        factory = Mock(return_value=[1])
        with patch.dict(os.environ, {"FEDOPS_CAMPAIGN_CONFIG": json.dumps({
                "serverEvaluation": {"enabled": True, "dataPath": "holdout-v1"}})}):
            policy = evaluation_policy(config)
            self.assertTrue(policy['enabled'])
            self.assertEqual(policy['data_root'], '/app/data/server-validation/holdout-v1')
            with patch('fedops.server.evaluation.validation_directory', return_value=Path(policy['data_root'])):
                prepare_validation_loader(config, factory)
            factory.assert_called_once_with(batch_size=8, data_root=policy['data_root'], download=False)
        factory.reset_mock()
        with patch.dict(os.environ, {"FEDOPS_CAMPAIGN_CONFIG": json.dumps({
                "serverEvaluation": {"enabled": False}})}):
            self.assertIsNone(prepare_validation_loader(config, factory))
        factory.assert_not_called()
        self.assertFalse(config.server_evaluation.enabled)
        self.assertNotIn('data_root', config.server_evaluation)

    def test_weighted_loss_accuracy_not_simple_average(self):
        summary = aggregate_client_evaluations([result('a', 10, 2., .5), result('b', 30, 4., 1.)])
        self.assertEqual(summary['gl_loss'], 3.5)
        self.assertEqual(summary['gl_accuracy'], .875)
        self.assertEqual(summary['evaluation_samples'], 40)
        self.assertNotIn('f1', summary)

    def test_missing_accuracy_has_separate_denominator(self):
        summary = aggregate_client_evaluations([result('a', 10, 2., .5), result('b', 30, 4.)])
        self.assertEqual(summary['accuracy_samples'], 10)
        self.assertEqual(summary['gl_accuracy'], .5)

    def test_invalid_and_duplicate_results_are_not_counted(self):
        summary = aggregate_client_evaluations([result('a', 10, 2., .5), result('a', 10, 2., .5),
                                               result('b', 0, 1.), result('c', 10, float('nan'))], 1)
        self.assertEqual(summary['evaluation_samples'], 10)
        self.assertEqual(summary['evaluation_status'], 'partial')
        self.assertEqual(summary['evaluation_failures'], 4)

    def test_empty_results_are_not_fake_zeros(self):
        summary = aggregate_client_evaluations([], 2)
        self.assertIsNone(summary['gl_loss'])
        self.assertIsNone(summary['gl_accuracy'])
        self.assertEqual(summary['evaluation_status'], 'not_evaluated')

    def test_reporting_strategy_preserves_fit_and_reports_round(self):
        delegate = FedAvg()
        report = Mock()
        strategy = ReportingStrategy(delegate, report)
        loss, metrics = strategy.aggregate_evaluate(3, [result('a', 10, 2., .5), result('b', 30, 4., 1.)], [])
        self.assertEqual((loss, metrics), (3.5, {"accuracy": .875}))
        self.assertEqual(report.call_args.args[0], 3)
        delegate.aggregate_fit = Mock(return_value=(None, {}))
        strategy.aggregate_fit(3, [], [])
        delegate.aggregate_fit.assert_called_once_with(3, [], [])

    def test_failure_rejection_policy_is_preserved(self):
        strategy = ReportingStrategy(FedAvg(accept_failures=False), Mock())
        self.assertEqual(strategy.aggregate_evaluate(1, [result('a', 10, 2., .5)], [RuntimeError()]), (None, {}))

    def test_invalid_results_cannot_crash_delegate_and_explicit_metrics_survive(self):
        report = Mock()
        delegate = FedAvg(evaluate_metrics_aggregation_fn=lambda _: {"f1": .7})
        strategy = ReportingStrategy(delegate, report)
        self.assertEqual(strategy.aggregate_evaluate(1, [result('a', 0, 2.)], []), (None, {}))
        loss, metrics = strategy.aggregate_evaluate(2, [result('a', 10, 2., .5), result('b', -10, 3.)], [])
        self.assertEqual(loss, 2.)
        self.assertEqual(metrics, {"accuracy": .5, "f1": .7})
        self.assertEqual(report.call_args.args[1]['additional_metrics'], {"f1": .7})

    def test_model_saved_without_server_evaluation(self):
        import numpy as np
        import torch
        from fedops.server.app import FLServer
        server = object.__new__(FLServer)
        server.model_type = 'Pytorch'
        server.server = SimpleNamespace(gl_model_v=1, round=1)
        server.client_evaluation = True
        server.test_torch = Mock(side_effect=AssertionError('server eval must not run'))
        model = torch.nn.Linear(1, 1)
        params = [np.ones_like(value.detach().numpy()) for value in model.state_dict().values()]
        with patch('torch.save') as save:
            self.assertIsNone(server.get_eval_fn(model, 'test')(1, params, {}))
            save.assert_called_once()
        self.assertTrue(all(torch.all(value == 1) for value in model.state_dict().values()))

    def test_two_round_fl_engine_saves_and_reports_without_validation(self):
        import numpy as np
        import torch
        from flwr.common import FitRes, parameters_to_ndarrays
        from flwr.server import Server
        from flwr.server.client_manager import SimpleClientManager
        from flwr.server.client_proxy import ClientProxy
        from fedops.server.app import FLServer

        received = []
        class LocalProxy(ClientProxy):
            def get_properties(self, *args, **kwargs):
                raise AssertionError('not used')
            def get_parameters(self, *args, **kwargs):
                raise AssertionError('explicit initial parameters required')
            def reconnect(self, *args, **kwargs):
                return None
            def fit(self, ins, **kwargs):
                params = [array + 1 for array in parameters_to_ndarrays(ins.parameters)]
                return FitRes(Status(Code.OK, ''), ndarrays_to_parameters(params), 10, {})
            def evaluate(self, ins, group_id=None, **kwargs):
                received.append((self.cid, group_id, parameters_to_ndarrays(ins.parameters)[0].copy()))
                return result(self.cid, 10 if self.cid == 'a' else 30, 2. if self.cid == 'a' else 4., .5 if self.cid == 'a' else 1.)[1]

        runtime = object.__new__(FLServer)
        runtime.model_type = 'Pytorch'
        runtime.server = SimpleNamespace(gl_model_v=4, round=0, start_by_round=0)
        runtime.client_evaluation = True
        runtime.task_id = 'test-task'
        runtime.batch_size = 8
        runtime.local_epochs = 1
        runtime.num_rounds = 2
        model = torch.nn.Linear(1, 1)
        initial = [np.zeros_like(value.detach().numpy()) for value in model.state_dict().values()]
        strategy = FedAvg(initial_parameters=ndarrays_to_parameters(initial), min_fit_clients=2,
                          min_evaluate_clients=2, min_available_clients=2,
                          evaluate_fn=runtime.get_eval_fn(model, 'integration'),
                          on_fit_config_fn=runtime.fit_config)
        manager = SimpleClientManager()
        manager.register(LocalProxy('a'))
        manager.register(LocalProxy('b'))
        engine = Server(client_manager=manager, strategy=ReportingStrategy(strategy, runtime.report_client_evaluation))
        save_to_disk = torch.save
        with tempfile.TemporaryDirectory() as directory, patch('torch.save', side_effect=lambda value, filename: save_to_disk(value, Path(directory) / filename)) as save, patch('fedops.server.app.server_api.ServerAPI') as api:
            with patch.dict(os.environ, {'FEDOPS_CAMPAIGN_RUN_ID': 'campaign-test'}):
                history, _ = engine.fit(num_rounds=2, timeout=10)
            self.assertEqual(save.call_count, 3)  # initial snapshot + both rounds
            saved = torch.load(Path(directory) / 'integration_gl_model_V4.pth', weights_only=True)
            self.assertTrue(all(torch.allclose(value, torch.full_like(value, 2.)) for value in saved.values()))
            payloads = [json.loads(call.args[0]) for call in api.return_value.put_gl_model_evaluation.call_args_list]
        self.assertEqual([p['round'] for p in payloads], [1, 2])
        self.assertTrue(all(p['gl_model_v'] == 4 and p['campaign_run_id'] == 'campaign-test' for p in payloads))
        self.assertEqual(history.losses_distributed, [(1, 3.5), (2, 3.5)])
        self.assertEqual(len(received), 4)
        for _, round_number, parameters in received:
            self.assertTrue(np.allclose(parameters, round_number))

    def test_server_evaluation_reports_source_and_rejects_nonfinite_metrics(self):
        import numpy as np
        import torch
        import time
        from fedops.server.app import FLServer
        runtime = object.__new__(FLServer)
        runtime.model_type = 'Pytorch'
        runtime.server = SimpleNamespace(gl_model_v=2, round=1, start_by_round=time.time())
        runtime.client_evaluation = False
        runtime.evaluation_policy = {'enabled': True}
        runtime.cfg = {}
        runtime.gl_val_loader = [1]
        runtime.test_torch = Mock(return_value=(2., .75, None))
        runtime.task_id = 'test'
        runtime.is_cluster = False
        model = torch.nn.Linear(1, 1)
        params = [np.zeros_like(value.detach().numpy()) for value in model.state_dict().values()]
        with patch('torch.save'), patch('fedops.server.app.server_api.ServerAPI') as api:
            callback = runtime.get_eval_fn(model, 'test')
            self.assertEqual(callback(1, params, {}), (2., {'accuracy': .75}))
            payload = json.loads(api.return_value.put_gl_model_evaluation.call_args.args[0])
            self.assertEqual(payload['evaluation_source'], 'server_validation')
            self.assertEqual(payload['evaluation_status'], 'evaluated')
            runtime.test_torch.return_value = (float('nan'), .75, None)
            with self.assertRaisesRegex(ValueError, 'non-finite'):
                callback(2, params, {})


if __name__ == '__main__':
    unittest.main()
