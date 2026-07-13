# MuSE-Toolbox Project Rules & Agent Directives

When operating in this workspace, you MUST adhere to the following rules at all times. These rules ensure that the codebase remains clean, highly modular, and consistent across all development sessions.

## 1. Master Context
**ALWAYS read `CURRENT_STATUS.md` at the root of the repository BEFORE taking any action.** 
This file contains the persistent context, architectural decisions, past accomplishments, and the immediate roadmap of the `MuSE-Toolbox` project. It replaces any older `docs/agent_history/` directory.

## 2. The 5-Point SOP
Any time you edit an existing file or create a new Python file, you must strictly apply the 5-Point Standard Operating Procedure:
1. **Absolute Imports:** Never use deep relative imports (e.g., `from ... import`). Always use absolute paths originating from the package root: `from muse_toolbox.module import...`.
2. **Professional Logging:** Never use `print()`. Always use Python's built-in `logging` module (`import logging; log = logging.getLogger(__name__)`).
3. **Strict Type-Hinting:** Every function and method signature must have complete Python type-hints (using the `typing` module, e.g., `Optional`, `Union`, `torch.Tensor`).
4. **Google-Style Docstrings:** All classes, methods, and functions must be documented strictly using Google-style docstrings.
5. **Clean Interfaces:** Estimators must inherit from their respective base classes (e.g., `BaseRTFestimator` or `BaseSourceCountEstimator`), which mandates a `get_config()` method and a `forward_()` pass. Never use deprecated PyTorch patterns (like `Variable`).

## 3. Architectural Boundaries
- **Generic Components:** Neural network blocks (`causal_conv1d.py`, etc.) and generic feature extractors (`IPD`, `WGMSC`, etc.) belong strictly in `src/muse_toolbox/models/components/`.
- **Domain-Specific Estimators:** RTF estimators belong in `models/rtf_estimation/estimators/`. Source Counting estimators belong in `models/source_counting/estimators/`.
- **Configurations:** The project uses **Hydra**. Never hardcode parameters in Python files. Define them in the modular YAML files inside `configs/` and instantiate them using `hydra.utils.instantiate()`.
- **Data Modules:** Avoid monolithic files. Data logic is decoupled into raw DBs, scenario generators, precomputed sets, and PyTorch Lightning DataModules.

## 4. Execution Protocol
- When refactoring large sections of code, **work file-by-file**. Present the completed file (or significant chunks) to the user before moving on to the next.
- Before executing commands that modify multiple files simultaneously, always seek explicit user approval via a plan.
