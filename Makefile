.PHONY: install-dev lint test typecheck docs precommit setup-paguro-dev clean

install-dev:
	@uv pip install -e .[test,lint,typing,docs]
	@uv pip install pre-commit

setup-paguro-dev: install-dev
	pre-commit install
	pre-commit autoupdate

test:
	pytest --cov=src --cov-report=term-missing

lint:
	ruff check src

typecheck:
	mypy src

docs:
	sphinx-build -v -b html docs/ docs/build/html

precommit:
	pre-commit run --all-files

clean:
	rm -rf .pytest_cache .mypy_cache docs/build dist *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf "{}" +
