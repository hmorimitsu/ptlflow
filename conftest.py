# =============================================================================
# Copyright 2026 Henrique Morimitsu
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

"""Global pytest configuration for the PTLFlow tests."""

import os

# The tests run many small model inferences, mostly on CPU. On machines with
# many CPU cores, letting each op use all of the cores causes heavy thread
# oversubscription, which can slow the tests down by orders of magnitude.
# Capping the number of threads avoids that. setdefault is used so that
# explicit values configured by the caller are respected. This must be set
# before torch is imported by the test modules.
_max_threads = str(min(8, os.cpu_count() or 1))
os.environ.setdefault("OMP_NUM_THREADS", _max_threads)
os.environ.setdefault("MKL_NUM_THREADS", _max_threads)
