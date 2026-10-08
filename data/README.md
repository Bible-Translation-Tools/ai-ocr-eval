# Dataset

Put the handwritten images here, each with a ground-truth `.txt` file of the same name.
Everything in this folder except this README is git-ignored, so the dataset stays on your machine.

```
data/
  page_001.jpg   page_001.txt
  page_002.png   page_002.txt
  letters/a.jpg  letters/a.txt     # subfolders are fine; the image id becomes "letters/a"
```

- **Images:** `.jpg`, `.jpeg`, `.png`, `.webp`, `.gif`. Images without a matching `.txt` are skipped with a warning.
- **Ground truth:** UTF-8 text of exactly what the page says. Separate paragraphs with one blank line;
  line breaks inside a paragraph don't matter (they are ignored when scoring).
- Include everything the model is expected to transcribe, **including titles**, and leave out
  crossed-out words. This matches the default prompt in `prompts/system.md`; if you change the
  prompt's rules, keep the ground truth consistent with them.

Run `ocr-eval check` to see which image/text pairs were found. To use another folder, set `data_dir`
in `config.yaml`.
