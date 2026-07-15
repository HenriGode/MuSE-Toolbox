from pathlib import Path
import os

base_dir = Path("/data4/Henri/MuSE-Toolbox/src/muse_toolbox")

for folder in ["utils", "data"]:
    target_dir = base_dir / folder
    for p in target_dir.rglob("*.py"):
        content = p.read_text()
        
        # fix utilities
        content = content.replace("from utilities import", "from muse_toolbox.utils import")
        content = content.replace("import utilities", "import muse_toolbox.utils as utilities")
        
        # fix datasets_local
        content = content.replace("from datasets_local import", "from muse_toolbox.data import")
        content = content.replace("import datasets_local", "import muse_toolbox.data as datasets_local")
        
        # fix internal datasets_local references
        content = content.replace("from .base_dataset import BaseDataset", "from muse_toolbox.data.base_dataset import BaseDataset")
        content = content.replace("from .sourceDBs import", "from muse_toolbox.data.sourceDBs import")
        
        p.write_text(content)

print("Done fixing imports in utils and data!")
