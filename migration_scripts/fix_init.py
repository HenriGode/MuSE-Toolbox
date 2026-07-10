from pathlib import Path
import os

metrics_dir = Path("/data4/Henri/MuSE-Toolbox/src/muse_toolbox/metrics")

for folder in ['common', 'source_counting', 'rtf_estimation']:
    folder_path = metrics_dir / folder
    init_path = folder_path / "__init__.py"
    files = [f.stem for f in folder_path.glob("*.py") if f.name != "__init__.py"]
    
    # Simple strategy: just import everything
    lines = ['"""Metrics module."""\n']
    for f in files:
        lines.append(f"from .{f} import *")
    
    init_path.write_text("\n".join(lines) + "\n")

top_init = metrics_dir / "__init__.py"
top_init.write_text('"""Metrics for MuSE-Toolbox."""\n')

print("Created __init__.py files")
