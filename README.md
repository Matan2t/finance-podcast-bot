# Finance Podcast Bot

A bot that generates finance podcast episodes from SEC filings using Gemini AI and uploads them to YouTube.

## Environment Setup

This project uses **conda** for environment management.

### Activate the environment

```bash
conda activate finance-podcast-bot
```

### Create the environment (first-time setup)

If the environment doesn't exist yet:

```bash
conda create -n finance-podcast-bot python=3.12 -y
conda activate finance-podcast-bot
conda run -n finance-podcast-bot pip install -r requirements.txt
```

### Run without activating

You can run the bot without manually activating:

```bash
conda run -n finance-podcast-bot python main.py
```

## Running the Bot

1. Activate the environment: `conda activate finance-podcast-bot`
2. Set required environment variables (e.g. `GEMINI_KEY`)
3. Run: `python main.py`

## Earnings call parser

Run the transcript parser for all companies in `companies.json`:

```bash
python3 earningcall_parser.py --companies-json companies.json --all-companies
```
