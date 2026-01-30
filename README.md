# Portfolio Commentary Generator

A Streamlit application that generates AI-powered quarterly portfolio commentary for investment professionals.

## Features

- **Excel Import**: Upload Brinson attribution data from Excel files
- **AI Generation**: Generate multiple commentary variations using GPT-4
- **Review & Edit**: Review, edit, and approve generated commentary
- **Export**: Export to Word, plain text, CSV, or JSON

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Key

Create `.streamlit/secrets.toml`:

```toml
OPENAI_API_KEY = "sk-your-openai-api-key"
```

Or set environment variable:

```bash
export OPENAI_API_KEY="sk-your-openai-api-key"
```

### 3. Run the Application

```bash
streamlit run app.py
```

## Configuration

Environment variables (optional):

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | (required) | OpenAI API key |
| `LLM_MODEL` | `gpt-4o` | Model to use |
| `LLM_TEMPERATURE` | `0.7` | Generation temperature |
| `LLM_MAX_TOKENS` | `2000` | Max tokens per request |
| `MAX_HOLDINGS` | `5` | Holdings per type |
| `VARIATIONS_COUNT` | `3` | Variations per holding |
| `DEBUG` | `false` | Enable debug mode |
| `LOG_LEVEL` | `INFO` | Logging level |

## Usage

1. **Upload**: Upload Excel file with attribution data
2. **Select**: Choose holdings to generate commentary for
3. **Generate**: Click to generate AI commentary
4. **Review**: Review, edit, and approve each commentary
5. **Export**: Download as Word document or other formats

## Excel File Format

Required columns:
- `Ticker` - Stock ticker symbol
- `Total Effect` - Attribution effect (bps)

Optional columns:
- `Company Name` - Full company name
- `Average Weight` - Portfolio weight (%)
- `Selection Effect` - Selection attribution (bps)
- `Allocation Effect` - Allocation attribution (bps)

## Development

### Run Tests

```bash
pytest tests/ -v
```

### Project Structure

```
portfolio-commentary/
├── app.py                 # Main application
├── src/
│   ├── config.py         # Configuration
│   ├── models.py         # Data models
│   ├── ui/               # UI components
│   ├── parsers/          # Excel parsing
│   ├── data/             # Thesis registry
│   ├── generation/       # LLM client
│   ├── export/           # Word export
│   └── utils/            # Validators, logging
├── data/                  # Data files
├── tests/                 # Test suite
└── .streamlit/           # Streamlit config
```

## License

Proprietary - Internal Use Only
