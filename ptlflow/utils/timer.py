"""Simple utilities to measure elapsed time."""

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

import logging
import time
from typing import Union, Tuple

try:
    import torch
except NameError:
    torch = None


class Timer(object):
    """Utility to count the total elapsed time.

    Every time toc() is called, the elapsed time since the last tic() is added to the total time. Call reset() to zero the
    total time.

    Attributes
    ----------
    name : str
        A string name that will be printed to identify this timer.
    indent_level : int, default 0
        The level of indentation when printing this timer. Useful for creating better prints. E.g., inner parts may have a
        larger indentation.

    Examples
    --------
    >>> t1 = Timer('parent_op', 0)
    >>> t1.tic()
    >>> ...
    >>> t2 = Timer('inner_op', 1)
    >>> t2.tic()
    >>> ...
    >>> t2.toc()
    >>> t1.toc()
    >>> print(t1)
    parent_op: 2000.0 (2000.0) ms
    >>> print(t2)
      inner_op: 1000.0 (1000.0) ms
    """

    def __init__(
        self,
        name: str,
        indent_level: int = 0
    ) -> None:
        """Initialize the Timer.

        Parameters
        ----------
        name : str
            A string name that will be printed to identify this timer.
        indent_level : int, optional
            The level of indentation when printing this timer. Useful for creating better prints. E.g., inner parts may have a
            larger indentation.
        """
        self.name = name
        self.indent_level = indent_level
        self.reset()

        self.num_tocs = 0

    def reset(self) -> None:
        """Zero the total time counter."""
        self.total_time = 0.0

    def tic(self) -> None:
        """Start to count the elapsed time."""
        self.has_tic = True
        if torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start = time.perf_counter()

    def toc(self) -> None:
        """Count the elapsed time since the last tic() and add it to the total time."""
        if torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.end = time.perf_counter()
        assert self.has_tic, 'toc called without tic'
        self.total_time += self.end - self.start
        self.has_tic = False
        self.num_tocs += 1

    def mean(self) -> float:
        """Return the average time (total time divided by the number of tocs).

        Returns
        -------
        float
            The average time in milliseconds.
        """
        return self.total() / max(1, self.num_tocs)

    def total(self) -> float:
        """Return the total time since the last reset().

        Returns
        -------
        float
            The total time in milliseconds.
        """
        return self.total_time

    def __repr__(self) -> str:
        return f'{"  "*self.indent_level}{self.name}: {1000 * self.total():.1f} ({1000 * self.mean():.1f}) ms'

    def __str__(self) -> str:
        return self.__repr__()


class TimerManager(object):
    """Utility to handle multiple timers.

    Timers can be accessed using a dict-like call (see Usage below). The timers can be either printed to the default output
    or to a log file.

    Attributes
    ----------
    timers : dict[Union[str, tuple[str, int]], Timer]
        The timers to be managed. The dict key can be either a single string representing the name of the timer, or a tuple
        (name, indentation_level)
    log_id : str, default 'timer'
        A string representing the name of this manager.
    log_path : str, default 'timer_log.txt'
        Path to where the log file will be saved (if log is used).
    logger : logging.Logger
        A hander for the logger.

    Examples
    --------
    >>> tm = TimerManager()
    >>> tm['op1'].tic()
    >>> ...
    >>> tm['op1'].toc()
    >>> # You may pass a tuple (str, int) as a key, which will be interpreted
    >>> # as the (name, indent_level) for the timer (see Timer above):
    >>> tm[('op2', 1)].tic()
    >>> ...
    >>> tm['op2'].toc()
    >>> print(tm)  # Prints all timers to default output
    op1: 2000.0 (2000.0) ms
      op2: 1000.0 (1000.0) ms
    >>> tm.write_to_log('Some header message (optional)')  # Write timers to a log file

    See Also
    --------
    Timer : The timers that are managed.
    """

    def __init__(
        self,
        log_id: str = 'timer',
        log_path: str = 'timer_log.txt'
    ) -> None:
        """Initialize the TimerManager.

        Parameters
        ----------
        log_id : str, default 'timer'
            A string representing the name of this manager.
        log_path : str, default 'timer_log.txt'
            Path to where the log file will be saved (if log is used).
        """
        self.timers = {}
        self.log_id = log_id
        self.log_path = log_path
        self.logger = None

    def clear(self) -> None:
        """Remove all timers."""
        self.timers = {}

    def reset(self) -> None:
        """Restart the total time counter of all timers."""
        for _, t in self.timers.items():
            t.reset()

    def write_to_log(
        self,
        header: str = ''
    ) -> None:
        """Write the timers to the log file.

        Parameters
        ----------
        header : str
            An optional string to be added to the top of the log file.
        """
        if self.logger is None:
            self._init_logger()
        if len(header) > 0:
            self.logger.info(header)
        self.logger.info(self.__repr__())

    def _init_logger(self) -> None:
        self.logger = logging.getLogger(self.log_id)
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler(self.log_path, mode='w')
        fh.setLevel(logging.INFO)
        self.logger.addHandler(fh)

    def __getitem__(
        self,
        key: Union[str, Tuple[str, int]]
    ) -> None:
        indent_level = 0
        if isinstance(key, tuple) or isinstance(key, list):
            indent_level = key[1]
            key = key[0]
        if self.timers.get(key) is None:
            self.timers[key] = Timer(key, indent_level)
        return self.timers[key]

    def __repr__(self) -> str:
        ret = ''
        for _, t in self.timers.items():
            ret += t.__repr__() + '\n'
        return ret

    def __str__(self) -> str:
        return self.__repr__()
