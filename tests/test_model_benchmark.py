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

from pathlib import Path
import shutil

from jsonargparse import ArgumentParser
import ptlflow
import model_benchmark

TEST_MODEL = "raft_small"


def test_benchmark(tmp_path: Path) -> None:
    model_ref = ptlflow.get_model_reference(TEST_MODEL)

    model_parser = ArgumentParser(parents=[model_benchmark._init_parser()])
    model_parser.add_argument_group("model")
    model_parser.add_class_arguments(model_ref, "model.init_args")
    args = model_parser.parse_args([])
    args.model.class_path = f"{model_ref.__module__}.{model_ref.__qualname__}"

    args.num_samples = 1
    args.output_path = tmp_path

    model_benchmark.benchmark(args, None)

    shutil.rmtree(tmp_path)
