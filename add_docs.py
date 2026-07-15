import os

metrics_dir = 'src/muse_toolbox/metrics'
for root, _, files in os.walk(metrics_dir):
    for file in files:
        if not file.endswith('.py'): continue
        path = os.path.join(root, file)
        
        with open(path, 'r') as f:
            lines = f.readlines()
            
        new_lines = []
        for i, line in enumerate(lines):
            new_lines.append(line)
            if line.startswith('class '):
                class_name = line.split(' ')[1].split('(')[0].split(':')[0].strip()
                # Check if next line is a docstring
                if i + 1 < len(lines) and '"""' not in lines[i+1]:
                    docstring = f'    """{class_name} metric class."""\n'
                    new_lines.append(docstring)
                    
        with open(path, 'w') as f:
            f.writelines(new_lines)
