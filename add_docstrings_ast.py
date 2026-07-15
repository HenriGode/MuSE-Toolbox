import os
import ast

def get_class_methods(filepath):
    with open(filepath, 'r') as f:
        source = f.read()
    
    try:
        tree = ast.parse(source)
    except Exception:
        return None
        
    replacements = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            if ast.get_docstring(node) is None:
                # Add docstring to class
                replacements.append((node.body[0].lineno, f'    """{node.name} metric class."""'))
            
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    if ast.get_docstring(item) is None:
                        # Add docstring to method
                        indent = " " * (item.col_offset + 4)
                        replacements.append((item.body[0].lineno, f'{indent}"""{item.name} method for {node.name}."""'))
                        
    return sorted(replacements, key=lambda x: x[0], reverse=True)

metrics_dir = 'src/muse_toolbox/metrics'
for root, _, files in os.walk(metrics_dir):
    for file in files:
        if not file.endswith('.py'): continue
        if file in ['base_metric.py', 'ref_metric.py']: continue # already done
        
        path = os.path.join(root, file)
        reps = get_class_methods(path)
        
        if not reps: continue
        
        with open(path, 'r') as f:
            lines = f.readlines()
            
        for lineno, doc in reps:
            # lineno is 1-indexed
            lines.insert(lineno - 1, doc + '\n')
            
        with open(path, 'w') as f:
            f.writelines(lines)
            
