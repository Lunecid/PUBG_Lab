PYTHON ?= python3
RUFF ?= ruff

.PHONY: structure format-check lint test-architecture test-synthetic wheel

structure:
	PYTHONPATH=src $(PYTHON) -m pubg_lab.repository.check_structure

format-check:
	$(RUFF) format --check src world_model movement_diffusion model_stgcn stdb network_analysis tools scripts tests/architecture

lint:
	$(RUFF) check src world_model movement_diffusion model_stgcn stdb network_analysis tools scripts tests/architecture

test-architecture:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src $(PYTHON) -m unittest discover -s tests/architecture -v

test-synthetic:
	env -u PG_PASSWORD -u PGPASSWORD -u DATABASE_URL -u PUBG_DB_PASSWORD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m pytest tests/architecture tests/world_model tests/movement_diffusion tests/corpus_quality tests/test_legacy_connection_config.py -q -p no:cacheprovider

wheel:
	$(PYTHON) -m pip wheel --no-index --no-deps --no-build-isolation --wheel-dir /tmp/pubg-lab-wheel .
