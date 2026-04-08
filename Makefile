.PHONY: install dev lint format typecheck test test-unit test-integration serve clean

install:
	pip install -r requirements.txt

dev:
	pip install -r requirements-dev.txt

lint:
	ruff check .

format:
	ruff format .

typecheck:
	mypy agent/ backend/

test: test-unit

test-unit:
	pytest tests/unit/ -q

test-integration:
	pytest tests/integration/ -m integration -q

serve:
	uvicorn backend.src.main:app --host 0.0.0.0 --port 8000 --reload

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache dist build *.egg-info
	rm -f backend_debug.log sse_stream.log
