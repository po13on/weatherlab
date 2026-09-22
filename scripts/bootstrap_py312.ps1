# Bootstrap Python 3.12 venv for LangChain / LlamaIndex extras.
# Core tests can still run on the system Python 3.8.

winget install Python.Python.3.12 --accept-package-agreements --accept-source-agreements
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install -U pip
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\python -m pytest -q
