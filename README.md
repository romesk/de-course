# Data Engineering by Capgemini | Course materials

## Quick Start

### Install the requirements

- Install the `poetry`: See [Installation](https://python-poetry.org/docs/#installation).
- _[Optional] If you do so, you can run without the `poetry run` command_
  - Create venv: `python3.12 -m venv .venv`
  - Activate venv: `source .venv/bin/activate`
- Install dependencies: `poetry install`
- Run script (use one of approaches):
  - - Activate poetry env: `poetry shell` or perform the optional step.
    - Run the scripts as usual: `python directory/src/script.py`
  - Run the script using `poetry run python directory/src/script.py`

## Tests

Run: `poetry run python -m unittest discover . `
