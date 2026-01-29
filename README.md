# Portfolio Commentary Generator

Streamlit app that transforms quarterly Excel performance data into human-quality investment commentary using GPT-5.

## Problem

Portfolio specialists spend several days each quarter writing 4-6 sentence summaries for top/bottom contributing stocks across multiple strategies. This tool reduces that to 1-2 hours of review and editing.

## What It Does

1. **Ingests** Excel workbook with Brinson attribution data (one tab per strategy)
2. **Identifies** top 5 contributors and bottom 5 detractors per strategy
3. **Retrieves** investment thesis context from a local registry
4. **Generates** 2-3 draft variations per holding using few-shot prompting with historical exemplars
5. **Presents** drafts for human review, selection, and editing
6. **Exports** approved commentary to Word document

## Project Status

See [PHASE_STATUS.md](PHASE_STATUS.md) for current build progress.

## Local Development

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/portfolio-commentary.git
cd portfolio-commentary

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure secrets
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit .streamlit/secrets.toml with your OpenAI API key

# Run
streamlit run app.py
```

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) — System design and component specs
- [PHASE_STATUS.md](PHASE_STATUS.md) — Build progress tracking
