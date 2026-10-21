# Repository instructions

Use Python 3.13, unittest, Black 24.8, and isort 5.13. Keep changes focused and preserve untracked files.

## Publication workflow

Commit and publish only within the user's authorized scope. Use repository-relative references and commands in publication text and generated metadata. Keep raw machine captures outside the tracked tree.

Install local hooks with `git config core.hooksPath .githooks`.

Before each commit, inspect the staged diff and exact commit message, then run:

```sh
python3 tools/check_public_text.py --staged --metadata
python3 tools/check_public_text.py --text-file commit-message.txt
```

Before each push, run `python3 tools/check_public_text.py --history --metadata`.

Before creating or editing a PR, comment, issue, release, or other publication, place the exact outgoing text in repository-relative temporary files and scan those files with `--text-file`. Send precisely the checked text. Never publish unreviewed shell output or attribution trailers. If a check cannot run, stop before publishing.

The guard blocks with a generic message; do not print matching content. Additional local aliases may be configured with repeated `git config --add privacy.identity VALUE` entries. Hooks supplement this workflow; do not bypass them.

Run guard tests with `python3 -m unittest tests.test_public_text -v`. Run the project's CPU checks with `CUDA_VISIBLE_DEVICES='' python3 -m unittest discover -v -f`; run relevant GPU checks for kernel changes.
