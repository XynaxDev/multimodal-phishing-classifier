# Contributing

Thanks for your interest in contributing to the Multimodal Phishing Detector project! The following notes explain how to contribute code, open issues, and submit pull requests.

## `Getting started`
- Fork the repository and create a feature branch from `main`:
  - `git checkout -b feature/my-feature`
- Keep your branch focused on a single change.

## `Code style`
- Python: follow PEP8. Use `black` and `flake8` if available.
- JavaScript/React: follow project linting rules; keep components small and well-documented.

## `Commit messages`
- Use clear, imperative commit messages. Example: `feat: add URL classifier wrapper` or `fix: handle missing image file`.

## `Pull requests`
- Open a PR against the `main` branch.
- Include a short description of what the change does, why it’s needed, and any relevant files changed.
- If your change affects model artifacts or dataset expectations, include instructions for reproducing or testing.

## `Testing`
- Make sure new code is covered by tests where applicable.
- Run local tests (the repository has local test scripts under `test/`) before submitting.

## `Model and data contributions`
- Do NOT commit large model weights or datasets directly to the repo. Use one of the following workflows:
  - Host large artifacts on cloud storage (S3, Azure Blob) and add a small script to download them.
  - Use Git LFS for model weights (if maintainers opt in).

## `Security and sensitive data`
- Do NOT include API keys, credentials, or other sensitive information in commits.
- If you discover a security issue, please open a private issue and contact the maintainer instead of posting it publicly.

## `Roadmap and issues`
- Use GitHub Issues to report bugs or request features.
- Provide steps to reproduce and minimal examples when possible.

## `Code of conduct`
- Be respectful and collaborative. If you'd like, add a reference to a formal code of conduct.

> **Thank you for contributing we appreciate your help!**
