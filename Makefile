UV ?= uv

.PHONY: help sync sync-all lock test lint format validate-config smoke clean

help:
	@echo "Targets:"
	@echo "  sync             Create/update .venv with the dev dependency group (default)"
	@echo "  sync-all         Sync dev plus all runtime extras (config/browser/train)"
	@echo "  lock             Refresh uv.lock without installing"
	@echo "  test             Run unit tests inside the uv-managed env"
	@echo "  lint             Run ruff checks"
	@echo "  format           Format with ruff and apply lint fixes"
	@echo "  validate-config  Validate the starter Breakout config"
	@echo "  smoke            Run config validation and unit tests"
	@echo "  clean            Remove .venv and build artifacts"

sync:
	$(UV) sync

sync-all:
	$(UV) sync --all-extras

lock:
	$(UV) lock

test:
	$(UV) run pytest

lint:
	$(UV) run ruff check src tests

format:
	$(UV) run ruff format src tests
	$(UV) run ruff check --fix src tests

validate-config:
	$(UV) run jepa-rl validate-config --config configs/games/breakout.yaml

smoke: validate-config test

clean:
	rm -rf .venv build dist *.egg-info src/*.egg-info
