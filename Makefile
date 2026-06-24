.PHONY: install download-data process-data train train-fast test test-cov lint format \
        docker-build docker-up docker-down docker-logs clean-data clean-models clean-all \
        notebook pipeline report

install:
	pip install -e ".[dev]"

download-data:
	python scripts/download_data.py

process-data:
	python -c "from src.data.pipeline import run_data_pipeline; run_data_pipeline()"

train: process-data
	python scripts/train_models.py --models all --n-trials 50

train-fast: process-data
	python scripts/train_models.py --models ridge --n-trials 10

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

lint:
	ruff check src/ tests/
	mypy src/ --ignore-missing-imports

format:
	ruff format src/ tests/

docker-build:
	docker build -t aus-housing-ml:latest .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f api

clean-data:
	rm -rf data/raw/ data/processed/

clean-models:
	find models/ -name "*.pkl" -delete
	find models/ -name "metadata.json" -delete

clean-all: clean-data clean-models
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
	rm -rf .pytest_cache/ htmlcov/ .mypy_cache/

notebook:
	jupyter lab notebooks/

report:
	python scripts/generate_report.py
	@echo "Report written to reports/evaluation_report.html"

pipeline: download-data process-data train test report
	@echo "Full pipeline complete."
