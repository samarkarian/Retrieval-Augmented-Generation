install:
	uv sync

run:
	uv run python3 -m src/__main__.py

debug:
	uv run python -m pdb -m src

clean:
	rm -rf .mypy_cache pycache src/__pycache__

lint:
	mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

	flake8 .