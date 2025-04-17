# =============================================================================
# Copyright 2021 Henrique Morimitsu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import sys

from ptlflow.utils.lightning.ptlflow_cli import PTLFlowCLI
from ptlflow.utils.lightning.ptlflow_checkpoint_connector import (
    _PTLFlowCheckpointConnector,
)
from ptlflow.utils.lightning.ptlflow_trainer import PTLFlowTrainer
from ptlflow.utils.registry import RegisteredModel

TEST_MODEL = "raft_small"


def test_cli_no_model() -> None:
    sys.argv = sys.argv[:1]
    PTLFlowCLI(
        model_class=None,
        subclass_mode_model=False,
        parser_kwargs={"parents": []},
        run=False,
        parse_only=False,
        auto_configure_optimizers=False,
    )


def test_cli() -> None:
    sys.argv = sys.argv[:1]
    sys.argv.extend(["--model", "rapidflow"])
    PTLFlowCLI(
        model_class=RegisteredModel,
        subclass_mode_model=True,
        parser_kwargs={"parents": []},
        run=False,
        parse_only=False,
        auto_configure_optimizers=False,
    )


def test_checkpoint_connector_and_trainer() -> None:
    _PTLFlowCheckpointConnector(PTLFlowTrainer())
