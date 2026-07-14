import ast
import importlib
import sys
import os
import re

# We will just map every name in math and dsp to its submodule
modules = [
    'muse_toolbox.utils.math.conversions',
    'muse_toolbox.utils.math.complex_angles',
    'muse_toolbox.utils.math.covariance',
    'muse_toolbox.utils.math.geometry',
    'muse_toolbox.utils.math.matrix_ops',
    'muse_toolbox.utils.math.stochastics',
    'muse_toolbox.utils.math.windowing',
    'muse_toolbox.utils.dsp.acoustic_simulation',
    'muse_toolbox.utils.dsp.beamforming',
    'muse_toolbox.utils.dsp.plotting',
    'muse_toolbox.utils.dsp.power',
    'muse_toolbox.utils.dsp.stats',
    'muse_toolbox.utils.dsp.transforms',
    'muse_toolbox.utils.dsp.vad_spp',
    'muse_toolbox.utils.dsp.whitening',
]

symbol_to_module = {}
for modname in modules:
    try:
        mod = importlib.import_module(modname)
        if hasattr(mod, '__all__'):
            for sym in mod.__all__:
                symbol_to_module[sym] = modname
        else:
            for sym in dir(mod):
                if not sym.startswith('_'):
                    symbol_to_module[sym] = modname
    except Exception as e:
        print(f"Error loading {modname}: {e}")

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
    
    # Now replace the wildcard block with explicit imports
    import_lines = []
    for mod in sorted(module_to_used.keys()):
        syms = ", ".join(sorted(module_to_used[mod]))
        # use relative imports
        rel_mod = mod.replace('muse_toolbox.utils.', '.')
        import_lines.append(f"from {rel_mod} import {syms}")
    
    # regex to remove old math and dsp wildcard imports
    new_source = source
    import_block = "from .math.conversions import *\nfrom .math.complex_angles import *\nfrom .math.covariance import *\nfrom .math.geometry import *\nfrom .math.matrix_ops import *\nfrom .math.stochastics import *\nfrom .math.windowing import *\n"
    import_block += "from .dsp.acoustic_simulation import *\nfrom .dsp.beamforming import *\nfrom .dsp.plotting import *\nfrom .dsp.power import *\nfrom .dsp.stats import *\nfrom .dsp.transforms import *\nfrom .dsp.vad_spp import *\nfrom .dsp.whitening import *\n"
    
    # we will just replace matching lines
    lines = new_source.split('\n')
    filtered_lines = []
    for line in lines:
        if re.match(r'^from \.math\.[a-z_]+ import \*', line) or re.match(r'^from \.dsp\.[a-z_]+ import \*', line):
            continue
        filtered_lines.append(line)
    
    # insert new imports where the first removed import was
    # wait, just put them before from .tensor_ops import *
    new_source = '\n'.join(filtered_lines)
    
    new_imports_str = '\n'.join(import_lines)
    new_source = new_source.replace("from .tensor_ops import *", new_imports_str + "\nfrom .tensor_ops import *")
    
    with open(filename, 'w') as f:
        f.write(new_source)

print("Done")
