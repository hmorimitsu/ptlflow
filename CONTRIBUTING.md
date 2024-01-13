# Contributing to PTLFLow

All kinds of contributions are welcome, including but not limited to the following:

- Fixes of any type, either to the code or the docs.
- New optical flow models.
- Support for other datasets.
- Finding parameters that successfully train some of the available models.

## How to contribute

1. Uninstall existing ptlflow packages

```bash
pip uninstall ptlflow
```

2. Fork the ptlflow repository and clone it to your working machine.

3. Checkout a new branch with a meaning name, for example.

```bash
git checkout -b feat/foo_feature
# or
git checkout -b fix/foo_bug
```

4. Commit your changes to your fork.

5. Create a Pull Request on GitHub to merge your fork to ptlflow. Make sure that the fork is up to date with the main branch of ptlflow.

Note: If you plan to develop something that involve large changes, it is encouraged to open an issue for discussion first.

## Code style

The code should be compatible with the following library versions:

- [Python](https://www.python.org/) >= 3.8
- [PyTorch](https://pytorch.org/) >= 1.8
- [PyTorch Lightning](https://www.pytorchlightning.ai/) == 1.9.X

When writing your code, please adhere to these guidelines:

- Use meaningful names for variables, functions, and classes.

- Always use [type hints](https://docs.python.org/3.6/library/typing.html). These make it much easier to understand what is expect to come in or out of each function.
    - Please note that we use Python >= 3.6 as the minimum requirement. Therefore, you cannot use built-in collection types such as `list` and `dict`, which were [introduced only in Python 3.9](https://docs.python.org/3/whatsnew/3.9.html#type-hinting-generics-in-standard-collections). Instead, you have to use their equivalents imports from `typing`, such as `List` and `Dict`.

- Write [docstrings](https://www.python.org/dev/peps/pep-0257/) for all public methods, functions, classes, and modules.
    - We adopt the [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) standard for docstrings.

- Use [f-strings](https://realpython.com/python-f-strings/) to format your strings.

- Write small incremental changes.

    - Create a commit for each meaningful, small change you make. Do not just create a huge commit with everything, since it will be much harder to understand and to revert small bugs.

- Add tests (for [pytest](https://docs.pytest.org/)) when you develop new code.

    - Tests are stored in the `tests` folder. The folder structure should mimic the structure of the source code.

- Consider using `logging` instead of `print` whenever possible (read about [logging](https://docs.python.org/3/howto/logging.html)). For each module that uses logging, you should call `config_logging` at the beginning to keep the logging format consistent across the platform.

```python
# File foo.py
import logging

from ptlflow.utils.utils import config_logging

config_logging()

# Your code here
```

- Use [black](https://pypi.org/project/black/) to format your code.

## Acknowledgements

These contribution guidelines are based on those from https://github.com/open-mmlab/mmcv and https://github.com/kornia/kornia.