# Mathlib Search Agent

A modular Mathlib theorem search agent with multiple LLM response providers.

## Features

- **Modular architecture** with clear separation of concerns
- **Multiple response providers** (stdin, Ollama, Gemini)
- **Multi-round search** within single queries
- **Auto cleanup** of `<think>` tags from LLM output
- **Prompt logging** to files
- **Search budget management**

## Quick Start

```python
from mathlib_search_agent import MathlibSearchAgent, OllamaResponseProvider

# Create agent with Ollama
provider = OllamaResponseProvider(model="deepseek-r1")
agent = MathlibSearchAgent(response_provider=provider)

# Run interactive session
agent.run_interactive_session()
```

## Response Providers

### StdinResponseProvider
Manual testing via stdin input:
```python
provider = StdinResponseProvider()
```

### OllamaResponseProvider
Local Ollama API:
```python
provider = OllamaResponseProvider(model="deepseek-r1")
```

### GeminiResponseProvider
Google Gemini API:
```python
provider = GeminiResponseProvider(model="gemini-2.5-flash", api_key="your-api-key")
```

## Usage Examples

### Interactive Mode
```bash
python mathlib_search_agent.py
```
Choose provider:
- `1` for stdin (manual testing)
- `2` for Ollama (automatic)
- `3` for Gemini (automatic)

### Programmatic Usage
```python
from mathlib_search_agent import MathlibSearchAgent, GeminiResponseProvider

# Create agent
provider = GeminiResponseProvider(api_key="your-key")
agent = MathlibSearchAgent(response_provider=provider)

# Handle single query
result = agent.handle_query("Find theorems about coprime numbers")
print(result)

# Reset conversation
agent.reset_conversation()
```

## Configuration

### MathlibSearchAgent
- `max_searches`: Max searches per query (default: 15)
- `prompts_dir`: Directory for prompt logs (default: "prompts_log")

### Provider-specific
- **Ollama**: `model`, `base_url`
- **Gemini**: `model`, `api_key`

## Requirements

- `search_dataset.py` module
- `mathlib_agent_prompt.txt` system prompt file
- **Ollama**: Local Ollama service running
- **Gemini**: Google API key

## Installation

```bash
pip install -r requirements.txt
```

## Dataset Creation

Create the Mathlib dataset from Hugging Face:

```bash
python create_dataset.py
```

This script:
- Loads the `AI-MO/mathlib-declarations` dataset
- Filters out entries with missing signatures
- Saves to `selected_mathlib_dataset.jsonl` file

## File Structure

```
├── mathlib_search_agent.py      # Main module
├── mathlib_agent_prompt.txt     # System prompt
├── search_dataset.py            # Search utilities
├── create_dataset.py            # Dataset creation script
├── selected_mathlib_dataset.jsonl  # Processed dataset
├── prompts_log/                 # Prompt logs
└── requirements.txt             # Dependencies
```
