import os
import shutil
from pathlib import Path

def clear_python_cache(root_dir="."):
    """Clear all Python cache files and directories"""
    
    # Remove __pycache__ directories
    for pycache_dir in Path(root_dir).rglob("__pycache__"):
        try:
            shutil.rmtree(pycache_dir)
            print(f"✓ Deleted: {pycache_dir}")
        except Exception as e:
            print(f"✗ Error deleting {pycache_dir}: {e}")
    
    # Remove .pyc files
    for pyc_file in Path(root_dir).rglob("*.pyc"):
        try:
            pyc_file.unlink()
            print(f"✓ Deleted: {pyc_file}")
        except Exception as e:
            print(f"✗ Error deleting {pyc_file}: {e}")
    
    # Remove .pyo files
    for pyo_file in Path(root_dir).rglob("*.pyo"):
        try:
            pyo_file.unlink()
            print(f"✓ Deleted: {pyo_file}")
        except Exception as e:
            print(f"✗ Error deleting {pyo_file}: {e}")
    
    print("\n✅ Python cache cleared!")

if __name__ == "__main__":
    clear_python_cache()