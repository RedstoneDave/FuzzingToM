# ProacTest v0.2.0

Adds IN3 support, unified LLM interface, and a Streamlit + Plotly dashboard.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
# optional
pip install -r requirements.txt
```

## IN3
```bash
proactest-download-in3
proactest-convert-in3
```

## Experiments
```bash
proactest --config configs/toy.yaml
proactest --config configs/in3_openai.yaml
proactest --config configs/in3_anthropic.yaml
proactest --config configs/in3_openai_compatible.yaml
```

## Dashboard
```bash
streamlit run dashboard/app.py
```
