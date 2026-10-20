.PHONY: install test lint figures app-data app clean

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

# Rebuild the precomputed table the Streamlit app reads (needs the Kaggle data).
app-data:
	python -m f1.app_data

app:
	streamlit run app/streamlit_app.py

clean:
	rm -rf .coverage .pytest_cache .ruff_cache
