"""
Launcher for statistical analysis - executes the main analysis script from statis folder.
This script coordinates all statistical analyses using the modular framework.
"""

import os
import sys

def main():
    """Execute the main statistical analysis script from the statis subfolder."""
    # Get the path to the statis directory
    statis_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'statis')
    
    # Add statis directory to Python path
    sys.path.insert(0, statis_dir)
    
    # Import and execute the main analysis
    try:
        from run_statistical_analysis import main as run_analysis
        print("=== Launching Statistical Analysis ===")
        print(f"Executing from: {statis_dir}")
        run_analysis()
        print("=== Statistical Analysis Completed ===")
    except ImportError as e:
        print(f"Error importing analysis modules: {e}")
        print(f"Make sure all required files are in: {statis_dir}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during analysis execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
