import ast
import re

symbol_to_module = {}
for init_file, prefix in [('muse_toolbox/utils/math/__init__.py', 'muse_toolbox.utils.math.'), ('muse_toolbox/utils/dsp/__init__.py', 'muse_toolbox.utils.dsp.')]:
    with open(init_file, 'r') as f:
        tree = ast.parse(f.read())
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            mod_name = prefix + node.module.lstrip('.')
            for alias in node.names:
                symbol_to_module[alias.name] = mod_name

class NameFinder(ast.NodeVisitor):
    def __init__(self):
        self.names = set()
    def visit_Name(self, node):
        self.names.add(node.id)
        self.generic_visit(node)

for filename in ['muse_toolbox/utils/data_utils.py', 'muse_toolbox/utils/util_classes.py']:
    print(f"Processing {filename}...")
    with open(filename, 'r') as f:
        source = f.read()
    
    # parse AST
    try:
        tree = ast.parse(source)
    except Exception as e:
        print(f"Syntax error in {filename}: {e}")
        continue
    
    finder = NameFinder()
    finder.visit(tree)
    used_symbols = finder.names
    
    # group by module
    module_to_used = {}
    for sym in used_symbols:
        if sym in symbol_to_module:
            mod = symbol_to_module[sym]
            module_to_used.setdefault(mod, set()).add(sym)
    
    import_lines = []
    for mod in sorted(module_to_used.keys()):
        syms = ", ".join(sorted(module_to_used[mod]))
        rel_mod = mod.replace('muse_toolbox.utils.', '.')
        import_lines.append(f"from {rel_mod} import {syms}")
    
    lines = source.split('\n')
    filtered_lines = []
    for line in lines:
        if re.match(r'^from \.math\.[a-z_]+ import \*', line) or re.match(r'^from \.dsp\.[a-z_]+ import \*', line):
            continue
        filtered_lines.append(line)
    
    new_source = '\n'.join(filtered_lines)
    new_imports_str = '\n'.join(import_lines)
    new_source = new_source.replace("from .tensor_ops import *", new_imports_str + "\nfrom .tensor_ops import *")
    
    with open(filename, 'w') as f:
        f.write(new_source)

print("Done")
