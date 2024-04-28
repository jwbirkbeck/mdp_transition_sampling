For compatibility, minor changes to the `lpg_ftw` code have been made. This file documents all changes.

* import statement paths were changed throughout, from `import mjrl. ...` to `import src.lpg_ftw.mjrl. ...`

* The package `gym` was replaced throughout with `gymnasium`.

* Metaworld's v1 environments have been replaced with v2, with associated changes to entry point string paths.

* Changes to LPG_FTW's gym_env.py file for compatibility with gymnasium:
    * `env.spec.timestep_limit` replaced by a constant 500, the fixed step limit for all Metaworld v2 tasks

* Deletion of python package `tabulate` and associated usage in print statement.

* Replaced unset variable `timestep_limit` with constant 500 (suitable for all Metaworld v2 envs). 