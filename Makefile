.PHONY: install test demo fast clean

install:
	python3 -m pip install -e ".[dev]"

test:
	python3 -m pytest -q

demo:
	python3 -m photocat demo --out runs/demo

fast:
	python3 -m photocat demo --fast --out runs/fast

clean:
	rm -rf runs data/*.csv data/*.sqlite data/*.jsonl .pytest_cache build dist *.egg-info
