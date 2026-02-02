# Contributing to DeepScan Framework

Thank you for your interest in contributing to the DeepScan Framework! This document provides guidelines and instructions for contributing.

## How to Contribute

### Reporting Issues

If you find a bug or have a feature request, please open an issue on GitHub with:
- A clear description of the problem or feature
- Steps to reproduce (for bugs)
- Expected vs actual behavior
- Your environment (Python version, OS, etc.)

### Contributing Code

1. **Fork the repository** and create a new branch for your changes
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the coding standards below

3. **Test your changes**
   ```bash
   # Install in development mode
   pip install -e ".[dev]"
   
   # Run tests
   pytest
   
   # Run linting
   black deepscan/
   flake8 deepscan/
   mypy deepscan/
   ```

4. **Commit your changes** with clear, descriptive commit messages
   ```bash
   git commit -m "Add feature: description of what you added"
   ```

5. **Push to your fork** and open a Pull Request

### Pull Request Guidelines

- Provide a clear description of what your PR does
- Reference any related issues
- Ensure all tests pass
- Update documentation if needed
- Follow the existing code style

## Code Style

### Python Style

- Follow PEP 8 style guidelines
- Use `black` for code formatting (line length: 100)
- Type hints are encouraged but not required for all functions
- Use descriptive variable and function names

### Project Structure

- **Models**: Add new model runners in `deepscan/models/`
- **Evaluators**: Add new evaluators in `deepscan/evaluators/`
- **Datasets**: Add new dataset loaders in `deepscan/datasets/`
- **Summarizers**: Add new summarizers in `deepscan/summarizers/`

### Adding New Components

#### Adding a New Model Runner

1. Create a new file in `deepscan/models/` (e.g., `my_model.py`)
2. Implement a class inheriting from `BaseModelRunner`
3. Register the model in the model registry:
   ```python
   from deepscan.registry.model_registry import get_model_registry
   
   @get_model_registry().register_model("my_model")
   def create_my_model(**kwargs):
       # Your model creation logic
       return MyModelRunner(...)
   ```
4. Import and register in `deepscan/models/__init__.py`

#### Adding a New Evaluator

1. Create a new file in `deepscan/evaluators/` (e.g., `my_evaluator.py`)
2. Implement a class inheriting from `BaseEvaluator`
3. Register the evaluator:
   ```python
   from deepscan.evaluators.registry import get_evaluator_registry
   
   @get_evaluator_registry().register_evaluator("my_evaluator")
   class MyEvaluator(BaseEvaluator):
       def evaluate(self, model, dataset, **kwargs):
           # Your evaluation logic
           return results
   ```
4. Add to `deepscan/evaluators/__init__.py` if needed

#### Adding a New Dataset

1. Create a new file in `deepscan/datasets/` (e.g., `my_dataset.py`)
2. Implement a dataset loader function
3. Register the dataset:
   ```python
   from deepscan.registry.dataset_registry import get_dataset_registry
   
   @get_dataset_registry().register_dataset("my_dataset")
   def load_my_dataset(**kwargs):
       # Your dataset loading logic
       return dataset
   ```
4. Add to `deepscan/datasets/__init__.py` if needed

## Testing

- Write tests for new features in the `tests/` directory
- Follow the existing test patterns
- Ensure tests pass with `pytest`
- Aim for good test coverage

## Documentation

- Update README.md if you add new features
- Add docstrings to new functions and classes
- Follow the existing documentation style
- Update example configs if you change configuration schemas

## Development Setup

```bash
# Clone the repository
git clone https://github.com/AI45Lab/DeepScan.git
cd DeepScan

# Install in development mode with all dependencies
pip install -e ".[dev,all]"

# Run tests
pytest

# Format code
black deepscan/

# Type checking
mypy deepscan/
```

## Questions?

If you have questions about contributing, please open an issue or contact the maintainers.

Thank you for contributing to DeepScan Framework!
