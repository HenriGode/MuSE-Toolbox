import ast

with open('/data4/Henri/MuSE-Toolbox/src/muse_toolbox/utils/__init__.py', 'r') as f:
    tree = ast.parse(f.read())

names = []
for node in tree.body:
    if isinstance(node, ast.ImportFrom):
        if node.module in ('math', 'dsp'):
            for alias in node.names:
                names.append(alias.name)

names = sorted(names)
all_str = "__all__ = [\n"
for name in names:
    all_str += f'    "{name}",\n'
all_str += "]\n"

with open('/data4/Henri/MuSE-Toolbox/src/muse_toolbox/utils/__init__.py', 'a') as f:
    f.write('\n' + all_str)
