# Test Suite

This directory contains comprehensive tests for the TIMER training framework.

## Test Structure

- `test_models.py`: Tests for model creation, configuration, and forward pass
- `test_data_processing.py`: Tests for data processing modules (datasets, S3 preprocessor)
- `test_training.py`: Tests for training modules (Trainer, FineTuneTrainer)
- `test_utils.py`: Tests for utility functions (parameter counting, model loading)
- `test_integration.py`: End-to-end integration tests for the complete training pipeline
- `run_all_tests.py`: Script to run all tests

## Running Tests

### Run All Tests

```bash
# Using pytest directly
pytest tests/ -v

# Using the test runner script
python tests/run_all_tests.py
```

### Run Specific Test Files

```bash
# Test models only
pytest tests/test_models.py -v

# Test data processing only
pytest tests/test_data_processing.py -v

# Test training only
pytest tests/test_training.py -v

# Test utilities only
pytest tests/test_utils.py -v

# Test integration only
pytest tests/test_integration.py -v
```

### Run Specific Test Classes or Functions

```bash
# Run a specific test class
pytest tests/test_models.py::TestTimerModel -v

# Run a specific test function
pytest tests/test_models.py::TestTimerModel::test_model_creation -v
```

### Run with Coverage

```bash
# Install pytest-cov if not already installed
pip install pytest-cov

# Run tests with coverage report
pytest tests/ --cov=. --cov-report=html

# View coverage report
# Open htmlcov/index.html in your browser
```

## Test Coverage

The test suite covers:

1. **Model Tests**:
   - Configuration creation and validation
   - Model instantiation
   - Forward pass (training and inference modes)
   - Parameter counting

2. **Data Processing Tests**:
   - Dataset creation and data loading
   - S3 preprocessor functionality
   - Sequence processing and sampling

3. **Training Tests**:
   - Trainer initialization
   - Training epoch execution
   - Validation
   - Fine-tuning trainer

4. **Utility Tests**:
   - Parameter counting
   - Model saving and loading

5. **Integration Tests**:
   - End-to-end training pipeline
   - Model save/load workflow
   - Training history and output files

## Requirements

All tests require:
- pytest >= 7.0.0
- pytest-cov >= 4.0.0 (optional, for coverage)

Install with:
```bash
pip install -r requirements.txt
```

## Notes

- Tests use temporary directories for output files
- Tests use CPU by default (no GPU required)
- Tests use small models and datasets for fast execution
- Some tests may take longer (integration tests)

