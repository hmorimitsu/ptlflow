from pathlib import Path
import shutil

import ptlflow
import model_benchmark

TEST_MODEL = "raft_small"


def test_benchmark(tmp_path: Path) -> None:
    parser = model_benchmark._init_parser()
    model_ref = ptlflow.get_model_reference(TEST_MODEL)
    parser = model_ref.add_model_specific_args(parser)
    args = parser.parse_args([TEST_MODEL])
    args.num_samples = 1
    args.output_path = tmp_path

    model_benchmark.benchmark(args, None)

    shutil.rmtree(tmp_path)
