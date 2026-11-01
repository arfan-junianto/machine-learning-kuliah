PYTHON ?= python3
VENV ?= .venv
VENV_PYTHON := $(VENV)/bin/python

.PHONY: setup train test app clean-artifacts

setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV_PYTHON) -m pip install -r requirements.txt

train:
	MPLCONFIGDIR=/tmp/ml-mpl $(VENV_PYTHON) train.py

test:
	MPLCONFIGDIR=/tmp/ml-mpl $(VENV_PYTHON) -m pytest -q

app:
	$(VENV)/bin/streamlit run app.py

clean-artifacts:
	$(VENV_PYTHON) -c "from pathlib import Path; import shutil; p=Path('artifacts'); shutil.rmtree(p) if p.is_dir() else None"

