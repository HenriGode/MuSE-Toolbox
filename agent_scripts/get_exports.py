import ast
import os
import glob

dsp_dir = "/data4/Henri/MuSE-Toolbox/src/muse_toolbox/utils/dsp"
files = glob.glob(os.path.join(dsp_dir, "*.py"))

exports = {}
for file in files:
    if file.endswith("__init__.py"):
        continue
    module_name = os.path.basename(file)[:-3]
    with open(file, "r") as f:
        tree = ast.parse(f.read())
    
    names = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            if not node.name.startswith("_"):
                names.append(node.name)
    if names:
        exports[module_name] = sorted(names)

import_lines = []
all_list = []
for mod in sorted(exports.keys()):
    names = exports[mod]
    all_list.extend(names)
    # create nicely wrapped import statements
    names_str = ", ".join(names)
    import_lines.append(f"from .{mod} import ({names_str})")

print("\n".join(import_lines))
print("\n__all__ = [")
for name in sorted(all_list):
    print(f'    "{name}",')
print("]")
