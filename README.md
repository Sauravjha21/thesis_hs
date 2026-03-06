# Thesis

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
uv sync
uv run pre-commit install
```

## Project Structure

```
data/           Raw, processed, and external datasets
notebooks/      Exploratory Jupyter notebooks
src/thesis/     Reusable Python package
scripts/        Pipeline entrypoints
tests/          pytest test suite
thesis/         LaTeX thesis document
results/        Generated figures and tables
```

## Workflow

```bash
# Run quality checks
uv run ruff format .
uv run ruff check --fix .
uv run ty check
uv run pytest

# Or all at once
uv run pre-commit run --all-files
```

## Notebooks

Notebooks live in `notebooks/`. Heavy logic belongs in `src/thesis/` — notebooks should call package functions.

## Writing

The LaTeX thesis is in `thesis/main.tex`. Compile with preferred LaTeX toolchain (e.g. `latexmk -pdf thesis/main.tex`).
