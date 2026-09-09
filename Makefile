.PHONY: help install test lint typecheck train score docker-build docker-up docker-down docker-logs docker-test clean

PYTHON ?= python3

help:
	@echo "PromoLift Development & Deployment Commands:"
	@echo "  make install       Install local package in editable mode with all extras"
	@echo "  make test          Run pytest suite with coverage"
	@echo "  make lint          Run Ruff linter"
	@echo "  make typecheck     Run Mypy static type checker"
	@echo "  make train         Train T-Learner and save production model artifact"
	@echo "  make score         Score sample campaign using batch CLI"
	@echo "  make docker-build  Build Docker images"
	@echo "  make docker-up     Start API, Dashboard, and Frontend services via Docker Compose"
	@echo "  make docker-down   Stop and remove Docker containers"
	@echo "  make docker-logs   Follow container logs"
	@echo "  make docker-test   Execute end-to-end container smoke test"

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[all]"

test:
	$(PYTHON) -m pytest tests/ --cov=promolift --cov-report=term-missing

lint:
	ruff check src/ tests/

typecheck:
	$(PYTHON) -m mypy src/promolift

train:
	$(PYTHON) -m promolift.cli train --output-dir models/production

score:
	$(PYTHON) -m promolift.cli score --campaign P001 --input data/customer_features.parquet --output outputs/targeting_list_sample.csv

docker-build:
	docker compose build

docker-up:
	docker compose up -d --build
	@echo "\nPromoLift is running!"
	@echo "  Frontend:   http://localhost:8080"
	@echo "  Dashboard:  http://localhost:8501"
	@echo "  API:        http://localhost:8000"
	@echo "  API Docs:   http://localhost:8000/docs"
	@echo "  Health:     http://localhost:8000/health"

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

docker-test:
	@echo "Running Docker smoke test..."
	@curl -s -f http://localhost:8000/health > /dev/null && echo "✓ /health endpoint OK" || (echo "✗ /health failed"; exit 1)
	@curl -s -f http://localhost:8000/model-info > /dev/null && echo "✓ /model-info endpoint OK" || (echo "✗ /model-info failed"; exit 1)
	@curl -s -f -X POST http://localhost:8000/predict \
		-H "Content-Type: application/json" \
		-d '{"campaign_id":"P001","customers":{"customer_id":"C001","recency_days":10,"frequency_30d":2,"monetary_90d":200,"total_spend":800,"total_visits":4,"total_items":10,"avg_basket_value":200,"promo_ratio":0.2,"customer_segment_code":2}}' > /dev/null && echo "✓ /predict endpoint OK" || (echo "✗ /predict failed"; exit 1)
	@curl -s -f http://localhost:8501/_stcore/health > /dev/null && echo "✓ Dashboard healthcheck OK" || (echo "✗ Dashboard healthcheck failed"; exit 1)
	@curl -s -f http://localhost:8080/nginx-health > /dev/null && echo "✓ Frontend (nginx) OK" || (echo "✗ Frontend healthcheck failed"; exit 1)
	@echo "All container smoke tests passed successfully!"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .coverage htmlcov/ dist/ build/
