from pathlib import Path

base_dir = Path("/data4/Henri/MuSE-Toolbox/src/muse_toolbox")

for folder in ["models"]:
    target_dir = base_dir / folder
    for p in target_dir.rglob("*.py"):
        content = p.read_text()
        
        # fix utilities
        content = content.replace("from utilities import", "from muse_toolbox.utils import")
        content = content.replace("import utilities", "import muse_toolbox.utils as utilities")
        
        # fix building_blocks
        content = content.replace("from building_blocks.", "from muse_toolbox.models.building_blocks.")
        content = content.replace("import building_blocks.", "import muse_toolbox.models.building_blocks.")
        
        # fix losses
        content = content.replace("from losses.", "from muse_toolbox.losses.")
        
        # fix metrics
        content = content.replace("from metrics.", "from muse_toolbox.metrics.")
        
        # fix base model
        content = content.replace("from .base_model import", "from muse_toolbox.models.common.base_model import")
        
        p.write_text(content)

print("Done fixing imports in models!")
