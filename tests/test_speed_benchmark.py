from pathlib import Path
import shutil

import ptlflow
import speed_benchmark

TEST_MODEL = 'raft_small'


def test_benchmark(tmp_path: Path) -> None:
    parser = speed_benchmark._init_parser()
    args = parser.parse_args([])
    args.model = TEST_MODEL
    args.num_samples = 1
    args.output_path = tmp_path

    speed_benchmark.benchmark(args)

    shutil.rmtree(tmp_path)
