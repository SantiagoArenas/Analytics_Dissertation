Python virtual environment (venv)

This project includes a virtual environment created at `.venv/` for reproducible Python development.

Quick start (macOS / zsh):

1. Activate the venv:

```bash
source .venv/bin/activate
```

2. Upgrade pip (optional):

```bash
python -m pip install --upgrade pip
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. To deactivate:

```bash
deactivate
```

Notes:
- The `.venv/` directory is intentionally local to the repository root. If you use a different shell or tool (e.g., conda), adjust activation accordingly.
- If you want the venv excluded from git, ensure `.gitignore` contains `.venv/` (we add one below if missing).