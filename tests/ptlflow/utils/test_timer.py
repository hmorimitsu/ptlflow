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

import time
from pathlib import Path

import pytest

from ptlflow.utils.timer import Timer, TimerManager


def test_timer_total_accumulates() -> None:
    timer = Timer("op")
    for _ in range(3):
        timer.tic()
        time.sleep(0.01)
        timer.toc()

    assert timer.num_tocs == 3
    assert timer.total_time > 0.02


def test_timer_mean() -> None:
    timer = Timer("op")
    for _ in range(4):
        timer.tic()
        time.sleep(0.005)
        timer.toc()

    # the mean must be the total divided by the number of tocs
    assert timer.mean() == pytest.approx(timer.total() / 4)


def test_timer_mean_single_toc() -> None:
    timer = Timer("op")
    timer.tic()
    time.sleep(0.005)
    timer.toc()
    assert timer.mean() == pytest.approx(timer.total())


def test_timer_toc_without_tic() -> None:
    timer = Timer("op")
    with pytest.raises(AssertionError):
        timer.toc()


def test_timer_toc_after_reset() -> None:
    timer = Timer("op")
    timer.tic()
    timer.reset()
    with pytest.raises(AssertionError):
        timer.toc()


def test_timer_reset() -> None:
    timer = Timer("op")
    timer.tic()
    time.sleep(0.005)
    timer.toc()
    assert timer.total_time > 0

    timer.reset()
    assert timer.total_time == 0.0
    assert timer.num_tocs == 0


def test_timer_reset_restarts_counters() -> None:
    timer = Timer("op")
    for _ in range(10):
        timer.tic()
        time.sleep(0.001)
        timer.toc()

    timer.reset()
    for _ in range(2):
        timer.tic()
        time.sleep(0.005)
        timer.toc()

    # the counters must restart after reset(), so the mean considers
    # only the 2 new tocs, and not the 10 old ones
    assert timer.num_tocs == 2
    assert timer.mean() == pytest.approx(timer.total() / 2)


def test_timer_repr() -> None:
    timer = Timer("op", 1)
    timer.tic()
    time.sleep(0.005)
    timer.toc()
    text = str(timer)
    assert "op" in text
    assert "ms" in text
    assert text.startswith("  ")  # indent level 1


def test_timer_manager_access() -> None:
    manager = TimerManager()

    timer1 = manager["op1"]
    timer2 = manager[("op2", 1)]

    assert isinstance(timer1, Timer)
    assert timer1.indent_level == 0
    assert timer2.indent_level == 1
    assert manager["op1"] is timer1  # the same timer is returned

    timer1.tic()
    time.sleep(0.005)
    timer1.toc()

    assert "op1" in str(manager)
    assert "op2" in str(manager)


def test_timer_manager_global_toc() -> None:
    manager = TimerManager()
    timer = manager["op"]
    timer.tic()
    time.sleep(0.005)
    timer.toc()

    manager.global_toc()
    assert manager.num_global_tocs == 1
    assert timer.num_global_tocs == 1

    # the mean now uses the number of global tocs
    timer.tic()
    time.sleep(0.005)
    timer.toc()
    assert timer.mean() == pytest.approx(timer.total() / 1)


def test_timer_manager_reset_and_clear() -> None:
    manager = TimerManager()
    timer = manager["op"]
    timer.tic()
    time.sleep(0.005)
    timer.toc()

    manager.reset()
    assert timer.total_time == 0.0

    manager.clear()
    assert manager.timers == {}


def test_timer_manager_reset_clears_global_tocs() -> None:
    manager = TimerManager()
    timer = manager["op"]
    for _ in range(5):
        timer.tic()
        time.sleep(0.001)
        timer.toc()
        manager.global_toc()

    manager.reset()
    assert manager.num_global_tocs == 0
    assert timer.num_global_tocs == 0

    # after the reset, without global_toc, the mean must fall back to
    # the timer's own number of tocs
    for _ in range(3):
        timer.tic()
        time.sleep(0.005)
        timer.toc()
    assert timer.mean() == pytest.approx(timer.total() / 3)


def test_timer_manager_write_to_log(tmp_path: Path) -> None:
    log_path = tmp_path / "timer_log.txt"
    manager = TimerManager(log_path=str(log_path))

    timer = manager["op"]
    timer.tic()
    time.sleep(0.005)
    timer.toc()

    manager.write_to_log("header message")
    assert log_path.exists()
    content = log_path.read_text()
    assert "header message" in content
    assert "op" in content
