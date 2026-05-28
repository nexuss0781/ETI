# Contributing to CogniArch

Thank you for your interest in contributing to CogniArch! This document provides guidelines and instructions for contributors.

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Welcome newcomers and help them learn
- Keep discussions professional and on-topic

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/cogniarch.git
   cd cogniarch
   ```
3. **Set up development environment**:
   ```bash
   pip install -r requirements.txt
   pip install -e ".[dev]"
   ```
4. **Create a branch** for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Guidelines

### Code Style

- Follow PEP 8 style guidelines
- Use type hints for all function signatures
- Write docstrings in Google style
- Keep functions focused and small (< 50 lines preferred)

### Testing

- Write tests for new functionality
- Ensure all existing tests pass:
  ```bash
  pytest tests/
  ```
- Maintain > 90% code coverage

### Documentation

- Update documentation for new features
- Include examples in docstrings
- Add entries to CHANGELOG.md

### Commit Messages

Follow conventional commits format:
```
feat: add EWC regularization strategy
fix: correct gradient computation in tangent bundle
docs: update GRAD API documentation
refactor: simplify parametric lens composition
test: add tests for sheaf constraint propagation
```

## Pull Request Process

1. **Ensure your branch is up to date**:
   ```bash
   git fetch origin
   git rebase origin/main
   ```

2. **Run tests and linting**:
   ```bash
   pytest tests/ --cov=cogniarch
   black cogniarch/
   flake8 cogniarch/
   mypy cogniarch/
   ```

3. **Push your branch**:
   ```bash
   git push origin feature/your-feature-name
   ```

4. **Open a Pull Request** on GitHub with:
   - Clear description of changes
   - Link to related issues
   - Screenshots/examples if applicable
   - Checklist of completed items

## Areas for Contribution

### High Priority

- Implement core GRAD modules (lens.py, trainer.py, tangent.py)
- Add continual learning strategies (EWC, SI, GEM implementations)
- Build curriculum engine components
- Create comprehensive test suite

### Medium Priority

- Add visualization tools
- Implement additional optimizers
- Create tutorial notebooks
- Benchmark against standard datasets

### Low Priority

- Performance optimizations
- Additional documentation
- Example scripts
- Integration with other frameworks

## Questions?

- Open an issue for bugs or feature requests
- Use GitHub Discussions for general questions
- Contact maintainers for sensitive matters

## Recognition

Contributors will be acknowledged in:
- CHANGELOG.md
- README.md contributors section
- Annual contributor highlights

Thank you for making CogniArch better!
