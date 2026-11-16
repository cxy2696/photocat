.PHONY: install test demo fast bench torch fetch verify clean

install:
	python3 -m pip install -e ".[dev]"

test:
	python3 -m pytest -q

demo:
	python3 -m photocat demo --out runs/scenario

fast:
	python3 -m photocat demo --fast --out runs/scenario

bench:
	python3 -m photocat bench --dataset all

torch:
	python3 -m photocat bench --dataset gap --backend torch --ensemble 4

fetch:
	python3 scripts/fetch_data.py

verify:
	python3 scripts/verify_provenance.py

clean:
	rm -rf runs .pytest_cache build dist *.egg-info
	rm -f data/external/ees_her.csv data/external/ees_all_lit_data.xlsx
