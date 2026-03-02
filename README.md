# thegeneric10.github.io

## TheGeneric Website
- Main site: `index.html` (TheGeneric ver-0.26.1.10)
- OranAI UI: `oranai.html` (OranAI area ver-0.26.0301)

## OranAI local backend (`oran-a1`)
A local Python backend is included in `oran_a1_server.py`.

### Features
- Local AI generation via **PyTorch + Transformers** (no ChatGPT/Gemini API).
- Multiple chats and message history persisted in SQLite (`.oranai/oranai.sqlite3`).
- Chat endpoint returns generation time in milliseconds.

### Run
```bash
python -m pip install -r requirements.txt
python oran_a1_server.py
```
Then open:
- `http://127.0.0.1:8000/` for TheGeneric
- `http://127.0.0.1:8000/oranai` for OranAI
