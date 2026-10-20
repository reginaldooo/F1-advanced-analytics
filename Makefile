.PHONY: install test lint figures clean

install:
	pip install -e ".[dev]"
	pre-commit install

test:
	pytest

lint:
	ruff check .

# Re-executes the notebook, which regenerates images/*.png from src/f1.
figures:
	jupyter nbconvert --to notebook --execute --inplace f1-advanced-analytics.ipynb
	nbstripout f1-advanced-analytics.ipynb

clean:
	rm -rf .coverage .pytest_cache .ruff_cache
