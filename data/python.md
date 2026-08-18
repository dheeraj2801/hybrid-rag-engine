# Python — Practical Guide for Projects

Python is a versatile, high-level language widely used for web services, data engineering, and machine learning. This note covers pragmatic tooling and best practices for small-to-medium projects.

Environment and tooling

- Versions: use Python 3.10+ where possible for language features and long-term support.
- Virtual environments: use `python -m venv .venv` or `pipx` for isolation. `poetry` or `pip-tools` are recommended for reproducible installs.
- Dependency files:
	- `pyproject.toml` for modern packaging (PEP 517/518). Use `tool.poetry` or `setuptools` sections.
	- `requirements.txt` for pinned deployments (`pip freeze` or `pip-compile`).

Type checking and formatting

- Use type hints (`typing`) and run `mypy` as part of CI for stronger guarantees.
- Format code with `black` and lint with `ruff`/`flake8`.

Testing

- Unit tests with `pytest` and fixtures. Keep tests fast and deterministic.
- Integration tests may use local services (Docker) and smaller datasets.

Packaging & distribution

- For libraries: publish to PyPI with `twine` or use `poetry publish`.
- For applications: build container images and use `pyproject`/`requirements` for dependency installation.

Concurrency

- Use `asyncio` for I/O-bound concurrency (web servers, network I/O) and threads/processes for CPU-bound work.
- `concurrent.futures.ProcessPoolExecutor` for CPU tasks when needed.

Performance tips

- Profile with `cProfile` or `pyinstrument` to find bottlenecks.
- Use native libraries (numpy, PyTorch) for heavy numeric workloads.
- Cache results when safe, and batch expensive operations.

Security

- Never commit secrets; use environment variables or secret managers.
- Keep dependencies up to date and scan for vulnerabilities.

Minimal example: virtualenv + test

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install pytest black ruff
# run tests
pytest -q
```

Helpful patterns

- Config: use `pydantic` / `pydantic-settings` for typed config objects loaded from `.env` or environment.
- Logging: use structured logging (`logging` + JSON formatter) for production observability.
- Scripts: place CLI and scripts under `scripts/` and call project internals to reuse business logic.

Further reading

- The Python Packaging Authority: https://packaging.python.org
- Effective Python, Two Scoops of Django, and real Python tutorials.
