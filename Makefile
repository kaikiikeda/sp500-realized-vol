.PHONY: data results test all
data:      ; python scripts/build_data.py
results:   ; python scripts/run_all.py
test:      ; python -m pytest -q tests
all: data test results
