install:
	uv sync

run:
	uv run python -m src

debug:
	uv run python -m pdb -m src

clean:
	rm -rf .mypy_cache
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} +

lint:
	uv run flake8 . --exclude=.venv,data,moulinette
	uv run mypy . --warn-return-any --warn-unused-ignores \
		--ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

.PHONY: install run debug clean lint
