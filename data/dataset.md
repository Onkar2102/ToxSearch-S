# Dataset

Shared ToxSearch-S prompt / run data used for papers and analysis:

[Google Drive folder](https://drive.google.com/drive/folders/1ELA6jUJznGwOfgKeQvWZLdHM0Ycnm1G-?usp=share_link)

## Local seed prompts

For evolution runs, the default seed file is `data/prompt.csv` (column **`questions`**). Override with `--seed-file`.

Live run outputs are written under `data/outputs/` (gitignored). Do not commit raw run directories; point analysis scripts at a fixed `--output-dir` or a copied paper run path instead.
