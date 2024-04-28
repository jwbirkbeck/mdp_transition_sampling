For compatibility, minor changes to the `mask_lrl` code have been made. This file documents all changes.

* PPO_agent.py used relative paths to import `network`, `component` and `BaseAgent` code. These were changed to absolute module paths.

* `import *` statements used in all files, were replaced with specific imports, or removed in `__init__` files.

* `import gym` was replaced with `import gymnasium as gym` to use the package supported by Farama Foundation

* The class `MNISTConvBody` in `network_bodies.py` was removed so that `torchrl` was not added as a project requirement from otherwise irrelevant code

* Replaced `terminated` with `terminated or truncated` for compatibility with gymnasium

* `np.asscalar` replaced with `item()`

* task labels wrapped in lists and then transformed backed to scalars where required to allow code to run