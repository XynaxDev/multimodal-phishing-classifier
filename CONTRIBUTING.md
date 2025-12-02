# Contributing

Thanks for your interest in contributing! Here's how to get started.

## Getting Started

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Keep changes focused on a single improvement

## Code Guidelines

**Python:**
- Follow PEP8 standards
- Use `black` and `isort` for formatting
- Run `flake8` for linting

**JavaScript/React:**
- Follow existing code style
- Keep components small and focused
- Add comments for complex logic

## Commit Messages

Use clear, descriptive commits:
- `feat: add new feature`
- `fix: resolve bug`
- `docs: update documentation`
- `refactor: improve code structure`

## Pull Requests

1. Open PR against `main` branch
2. Describe what changed and why
3. Link related issues if applicable
4. Ensure all tests pass

## Testing

- Add tests for new features
- Run existing tests before submitting
- Verify models work if changes affect ML code

## Important Rules

**DO NOT commit:**
- Model weights or large files (use Git LFS or cloud storage)
- Datasets or training data
- API keys or credentials
- `.env` files with secrets

**Security:**
If you find a security issue, please report it privately via GitHub issues rather than publicly.

## Questions?

- Open an issue for bugs or feature requests
- Provide steps to reproduce when reporting bugs
- Be respectful and collaborative

---

**Thank you for contributing!**