from pathlib import Path
import os

metrics_dir = Path("/data4/Henri/MuSE-Toolbox/src/muse_toolbox/metrics")

for p in metrics_dir.rglob("*.py"):
    content = p.read_text()
    # fix base_metric and ref_metric imports
    content = content.replace("from .base_metric import BaseMetric", "from muse_toolbox.metrics.common.base_metric import BaseMetric")
    content = content.replace("from .ref_metric import RefMetric", "from muse_toolbox.metrics.common.ref_metric import RefMetric")
    
    # fix utilities
    content = content.replace("from utilities import", "from muse_toolbox.utils import")
    
    p.write_text(content)
print("Done fixing imports!")
