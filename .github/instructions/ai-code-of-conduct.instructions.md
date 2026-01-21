---
applyTo: '**/*.py, **/*.ipynb'
---

# AI Project Python Code of Conduct

## 1. Clean Code & Environment

* **Structure:** Use modular functions/classes with single responsibilities. Document with Google-style docstrings.
* **Naming:** PEP 8 (snake_case for functions, CamelCase for classes). Use descriptive names.
* **Imports:** Group by Standard, Third-party, and Local. No wildcard imports.
* **Environment:** Use `uv`. Manage dependencies via `uv add`. Always run within `.venv`. Never edit `requirements.txt` or `pyproject.toml` directly.
* **Type Hints:** Use type annotations for all functions and methods.
* **Init Files:** Include main classes, data models and necessary methods in `__init__.py` for easy imports.

## 2. Configuration & Logging

* **Config:** Use `pydantic_settings` for type-safe configuration. Use a global `config.py` for shared paths and stage-specific configs (e.g., `data_config.py`) for pipeline parameters. Use `<module>/config.py` files for each module.
* **Logging:** Prefer `logging` over `print`. Track app state, errors, and hardware utilization (CPU/GPU) for heavy tasks.

## 3. ML/DL Coding Style

* **Modular Code Structure:** Separate data loading, preprocessing, model definition, training, and evaluation into distinct modules. All modules should provide a `class` or `function` interface for easy integration.
* **Data Handling:** Use `torch.utils.data.Dataset` and `DataLoader` for efficient data management. Follow industrial best practices for data augmentation and preprocessing.
* **Model Definition:** Define models using `torch.nn.Module`. Use clear naming conventions for layers and components. Use pre-trained models from `torchvision.models` when applicable.
