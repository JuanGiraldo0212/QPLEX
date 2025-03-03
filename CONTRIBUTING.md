```markdown
# Contributing to QPLEX

Thank you for considering contributing to QPLEX! This document outlines the process for contributing to the project.

## Development Environment Setup

1. Create and activate a virtual environment:
```python
python -m venv venv
source venv/bin/activate  # Linux/MacOS
.\venv\Scripts\activate   # Windows
```

2. Install dependencies:
```python
pip install -r requirements.txt
pip install -e .  # Install QPLEX in development mode
```

## Code Style and Documentation

- Follow [PEP 8](https://peps.python.org/pep-0008/) style guide
- All code must include docstrings following [PEP 257](https://peps.python.org/pep-0257/)
- Required documentation levels:
  - Module-level: Purpose and contents
  - Class-level: Functionality, attributes, usage
  - Method-level: Purpose, parameters, returns, raises

Example docstring format:

- Class-level

```python
    """
    One-line description.

    More elaborate description of the class' purpose.

    Attributes
    ----------
    name : type
        Description
    """
```

- Method-level

```python
def method_name(self, param1: type, param2: type) -> return_type:
    """Short description of method functionality.

    Detailed description of the method's purpose and behavior.

    Parameters
    ----------
    param1 : type
        Description of first parameter
        
    param2 : type
        Description of first parameter


    Returns
    -------
        Description of return value

    Raises
    ------
        ExceptionType: Description of when/why this exception occurs
    """
```

## Testing

- Write tests using pytest
- Write tests in a class-based format
- Run tests with: `pytest tests/`
- Minimum 80% branch coverage required

Example test structure:
```python
def test_method_name():
    """Test description."""
    # Arrange
    input_data = ...
    
    # Act
    result = method_name(input_data)
    
    # Assert
    assert result == expected_output
```

## Pull Request Process

1. Branch naming:
   - `feature/description` for new features
   - `bugfix/description` for bug fixes
   - `docs/description` for documentation updates

2. PR requirements:
   - Clear description of changes
   - Reference related issues
   - Update documentation
   - Add/update tests
   - Pass all CI checks

3. Merge strategy:
   - Squash and merge for cleaner history
   - Write clear, descriptive commit messages

## Questions?

Open an issue for support or clarification.
```