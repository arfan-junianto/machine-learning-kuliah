PYTHON ?= python3

data:
	$(PYTHON) download_data.py

train: data
	$(PYTHON) train.py

test:
	$(PYTHON) -m pytest -q

app:
	$(PYTHON) -m streamlit run app.py
