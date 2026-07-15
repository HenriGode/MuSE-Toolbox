import os
import re

metrics_dir = 'src/muse_toolbox/metrics'

for root, _, files in os.walk(metrics_dir):
    for file in files:
        if not file.endswith('.py'): continue
        path = os.path.join(root, file)
        
        with open(path, 'r') as f:
            content = f.read()

        # Fix print in pesq.py
        if file == 'pesq.py' and 'print(' in content:
            if 'import logging' not in content:
                content = content.replace('import torch', 'import torch\nimport logging\n\nlog = logging.getLogger(__name__)')
            content = content.replace('print(', 'log.warning(')
            
        # Fix print in save_audio.py (if any exist)
        if file == 'save_audio.py' and 'print(' in content:
            if 'import logging' not in content:
                content = content.replace('import torch', 'import torch\nimport logging\n\nlog = logging.getLogger(__name__)')
            content = content.replace('print(', 'log.info(')

        # Fix unused variable W in fwssnr.py
        if file == 'fwssnr.py':
            content = re.sub(r'W = torch\.tensor\(\s*\[[\s\d.,]*\],\s*device=ref\.device,\s*dtype=ref\.dtype,\s*\)', '', content)
            
        with open(path, 'w') as f:
            f.write(content)
