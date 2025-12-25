"""
Run all tests in the test suite
"""
import sys
from pathlib import Path
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

if __name__ == "__main__":
    # Get test directory
    test_dir = Path(__file__).parent
    
    # Run all tests
    exit_code = pytest.main([
        str(test_dir),
        "-v",
        "--tb=short",
        "--color=yes"
    ])
    
    sys.exit(exit_code)

