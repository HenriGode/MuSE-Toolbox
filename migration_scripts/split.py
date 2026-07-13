import os
from pathlib import Path

base_file = "/data4/Henri/MuSE-Toolbox/src/muse_toolbox/data/base_dataset.py"

with open(base_file, "r") as f:
    lines = f.readlines()

imports = lines[0:35]
config_str = "".join(lines[35:54])
db_str = "".join(lines[54:142])
generator_str = "".join(lines[142:720])
precomp_str = "".join(lines[720:882])
datamod_str = "".join(lines[882:])

imports_str = "".join(imports)

# 1. base_dbs.py
os.makedirs("/data4/Henri/MuSE-Toolbox/src/muse_toolbox/data/databases", exist_ok=True)
with open("/data4/Henri/MuSE-Toolbox/src/muse_toolbox/data/databases/base_dbs.py", "w") as f:
    f.write(imports_str + "\n" + db_str)
    
# 2. scenario_generator.py
with open("/data4/Henri/MuSE-Toolbox/src/muse_toolbox/data/scenario_generator.py", "w") as f:
    f.write(imports_str + "\n")
    f.write("from .databases.base_dbs import BaseSourceDB, BaseRIRsDB, BaseNoiseDB\n\n")
    f.write(config_str + "\n" + generator_str)

# 3. precomputed_dataset.py
with open("/data4/Henri/MuSE-Toolbox/src/muse_toolbox/data/precomputed_dataset.py", "w") as f:
    f.write(imports_str + "\n" + precomp_str)

# 4. base_datamodule.py
with open("/data4/Henri/MuSE-Toolbox/src/muse_toolbox/data/base_datamodule.py", "w") as f:
    f.write(imports_str + "\n")
    f.write("from .precomputed_dataset import PrecomputedDataset\n\n")
    f.write(datamod_str)

print("Split completed.")
