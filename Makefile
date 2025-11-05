.PHONY: help setup install clean test lint format data train evaluate all

help:
	@echo "Bankruptcy Prediction - Available Commands:"
	@echo "  make setup      - Create virtual environment"
	@echo "  make install    - Install dependencies"
	@echo "  make clean      - Remove generated files"
	@echo "  make test       - Run tests"
	@echo "  make lint       - Run linters"
	@echo "  make format     - Format code"
	@echo "  make data       - Download and process data"
	@echo "  make train      - Train models"
	@echo "  make evaluate   - Evaluate models"
	@echo "  make all        - Run full pipeline"

setup:
	python -m venv venv
	@echo "Virtual environment created. Activate with:"
	@echo "  Windows: venv\\Scripts\\activate"
	@echo "  Unix: source venv/bin/activate"

install:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build dist .pytest_cache .coverage htmlcov

test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

lint:
	flake8 src/ scripts/ --max-line-length=120
	mypy src/ --ignore-missing-imports

format:
	black src/ scripts/ tests/
	isort src/ scripts/ tests/

data:
	python scripts/download_data.py

train:
	python scripts/train.py

evaluate:
	python scripts/evaluate.py

all: data train evaluate
	@echo "Full pipeline completed!"