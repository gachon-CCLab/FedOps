import unittest

import numpy as np
import torch
from torch import nn

from fedops.client.parameter_contract import (
    describe_parameters,
    get_parameters,
    parameter_signature,
    set_parameters,
    verify_parameter_round_trip,
)


class ContractModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)
        self.bn = nn.BatchNorm1d(2)


class ParameterContractTest(unittest.TestCase):
    def test_pytorch_contract_preserves_fedops_bn_exclusion(self):
        model = ContractModel()
        descriptor = describe_parameters(model, "Pytorch")
        self.assertEqual([item["name"] for item in descriptor], ["linear.weight", "linear.bias"])
        self.assertEqual(len(get_parameters(model, "Pytorch")), 2)

    def test_set_and_get_use_the_same_payload(self):
        model = ContractModel()
        values = [
            np.full((2, 4), 3.0, dtype=np.float32),
            np.full((2,), 4.0, dtype=np.float32),
        ]
        original_bn = model.bn.weight.detach().clone()
        set_parameters(model, "Pytorch", values)
        restored = get_parameters(model, "Pytorch")
        self.assertTrue(all(np.array_equal(left, right) for left, right in zip(values, restored)))
        self.assertTrue(torch.equal(original_bn, model.bn.weight))

    def test_signature_has_no_parameter_values_and_round_trip_matches(self):
        model = ContractModel()
        signature = parameter_signature(model, "Pytorch")
        self.assertEqual(signature["tensorCount"], 2)
        self.assertEqual(len(signature["fingerprint"]), 64)
        result = verify_parameter_round_trip(model, ContractModel, "Pytorch")
        self.assertTrue(result["ok"])
        self.assertEqual(result["signature"]["fingerprint"], signature["fingerprint"])


if __name__ == "__main__":
    unittest.main()
