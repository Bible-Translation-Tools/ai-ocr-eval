# ocr-eval

Measure how well AI vision models read handwriting. For every image in `data/` with a same-name `.txt`
ground truth, `ocr-eval` sends the image to each configured model with your system and user prompts,
scores the output against the ground truth, and writes a self-contained `results/report.html`.

Providers: OpenAI (and any OpenAI-compatible endpoint), Mistral chat models, Mistral OCR, Gemini
and Anthropic.

## Setup

```bash
uv sync                      # Python 3.10+; installs httpx, pyyaml, rapidfuzz, python-dotenv
cp .env.example .env         # add the API keys you have
uv run ocr-eval check        # lists image/.txt pairs and which models are ready
```

Without uv (plain pip, Python 3.10+):

```bash
python3 -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt   # exact tested versions
pip install -e . --no-deps
ocr-eval check                # or: python -m ocr_eval check
```

`requirements.txt` is generated from `uv.lock`. After changing dependencies in `pyproject.toml`, regenerate it with
`uv export --no-hashes --no-dev --no-emit-project -o requirements.txt`.

With the venv active, drop the `uv run` prefix from the commands below.

## Run

```bash
uv run ocr-eval run                          # all enabled models in config.yaml
uv run ocr-eval run --only gpt-5 --limit 2   # one model, first 2 images
uv run ocr-eval run --image story_ru         # only images whose id starts with story_ru
uv run ocr-eval report                       # rebuild report from cached outputs, no API calls
uv run ocr-eval run --force                  # ignore the cache and call the APIs again
```

Model outputs are cached in `results/predictions/<model>/<image>.json`. A cached output is reused while the
model settings, prompts and image are unchanged; change any of them and the next `run` calls the API again
(`report` marks such outputs as *stale*). Failed requests are retried on the next `run`.

Outputs: `results/report.html`, `results/results.json` (everything, per image and per model) and
`results/results.csv` (one row per model × image).

## Dataset

Put image + same-name `.txt` pairs in `data/` (e.g. `page_001.jpg` + `page_001.txt`). The folder's
contents are git-ignored, so your dataset stays local. Format rules (paragraphs, titles, crossed-out
words) are in [`data/README.md`](data/README.md).

## Metrics

Both texts are normalized first: Unicode NFC, typographic variants unified (`normalization.char_map`:
curly → straight quotes, en dash → em dash, … → ...), and whitespace inside a paragraph collapsed, so
the handwritten line wraps never count as errors.

| Metric | What it tells you |
|---|---|
| **CER** | Character error rate: Levenshtein edits ÷ reference characters, micro-averaged over images. The main number. |
| **CER lenient** | CER ignoring case, punctuation and `lenient_char_map` variants (ё/е). Pure letter reading. The gap to CER is the cost of punctuation and case. |
| **WER** | Word error rate, punctuation ignored. |
| **Punctuation P/R/F1** | Per mark (`, . : — ' "` ...) from a character alignment: did the model write the right mark in the right place? |
| **Paragraph ¶ P/R/F1** | Paragraph breaks aligned the same way, plus paragraph counts. |
| **Character-class recall** | Share of letters, digits, punctuation and spaces reproduced. |
| **Case / diacritic errors** | Substitutions that differ only by case (a/A) or a diacritic (е/ё). |
| **Top confusions** | Most frequent wrong words and characters per model. |

The report also shows latency, token usage, cost (when `price` is set per model) and a word- or
character-level diff of every output against the ground truth.

## Config

See the comments in `config.yaml`. Per model you can set `temperature`, `max_tokens`, `image_detail`
(OpenAI), `extra_body` (merged into the request JSON, e.g. `reasoning_effort` or Gemini `thinkingConfig`),
`system_prompt` / `user_prompt` (for prompt A/B tests), `base_url` + `api_key_env` (OpenAI-compatible
endpoints), `concurrency`, `price` and `enabled`.
