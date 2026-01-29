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

## Workflow

```
Upload Excel → Parse Holdings → Generate Drafts → Review/Edit → Export Word
```

## Tech Stack

- **UI:** Streamlit
- **LLM:** OpenAI GPT-5.x (ChatGPT Enterprise compatible)
- **Excel:** openpyxl
- **Word:** python-docx
- **Deployment:** Streamlit Community Cloud

## Project Status

This project is being built in phases. See [PHASE_STATUS.md](PHASE_STATUS.md) for current progress.

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Scaffolding | Not started |
| 1 | Excel Parser | Not started |
| 2 | Thesis Registry | Not started |
| 3 | Exemplar Parser | Not started |
| 4 | Prompt Builder | Not started |
| 5 | LLM Client | Not started |
| 6 | Core UI | Not started |
| 7 | Review UI | Not started |
| 8 | Export | Not started |
| 9 | Polish | Not started |

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

# Configure secrets (copy template and add your API key)
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit .streamlit/secrets.toml with your OpenAI API key

# Run
streamlit run app.py
```

## Configuration

### Local Development
Create `.streamlit/secrets.toml`:
```toml
OPENAI_API_KEY = "sk-..."
OPENAI_BASE_URL = "https://api.openai.com/v1"  # Override for Enterprise
```

### Streamlit Cloud
Add secrets via the Streamlit Cloud dashboard under App Settings → Secrets.

## Directory Structure

```
portfolio-commentary/
├── app.py                    # Streamlit entry point
├── requirements.txt
├── README.md
├── ARCHITECTURE.md           # Detailed system design
├── PHASE_STATUS.md           # Build progress tracking
├── .streamlit/
│   ├── config.toml           # Streamlit theme
│   └── secrets.toml.example  # Secrets template
├── src/
│   ├── parsers/              # Excel and Word parsing
│   ├── data/                 # Thesis registry
│   ├── generation/           # Prompt building and LLM calls
│   └── export/               # Word document generation
├── data/
│   ├── thesis_registry.csv   # Investment thesis database
│   ├── exemplars/            # Parsed historical blurbs
│   └── synthetic/            # Demo/test data
└── tests/
```

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) — Detailed system design and component specs
- [PHASE_STATUS.md](PHASE_STATUS.md) — Current build status and phase history

## License

Internal use only. Not for distribution.
