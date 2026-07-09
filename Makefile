.PHONY: install install-dev test smoke samples build clean doctor

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

test:
	VIDMCP_SAM_BACKEND=mock pytest tests/unit -q

smoke:
	VIDMCP_SAM_BACKEND=mock python examples/cyberpunk_edit.py

samples:
	python scripts/generate_samples.py

doctor:
	vidmcp --doctor

build:
	python -m build && twine check dist/*

clean:
	rm -rf dist build *.egg-info .pytest_cache
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} + 2>/dev/null || true
