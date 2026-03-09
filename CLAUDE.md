# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Main rules
- use `uv` to run python in this directory
- use `uvx ruff` to lint the files
- update the README.md file after each task completion
- use `/pal:review` with `gpt-5.4-thinking` after each task completion

## Project Overview

Implementation of influence functions for edge edits in non-convex Graph Neural Networks (GNNs), based on the paper `2506.04694v2.pdf` included in the repo.

## Setup

- Python >=3.12 (pinned to 3.12 via `.python-version`)
- Uses **uv** as the package manager (`pyproject.toml`, no lock file yet)
- `uv sync` to install dependencies
- `uv run main.py` to run the main entry point

## Architecture

Early-stage project. Currently a single `main.py` entry point with no dependencies. The reference paper is in the repo root as `2506.04694v2.pdf`.
