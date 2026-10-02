"""Download Pi3's original published weights inside Docker, recording provenance."""
import hashlib
import json
from pathlib import Path
import time
import urllib.request

root = Path('/run/user/1016/pi3')
root.mkdir(parents=True, exist_ok=True)
url = 'https://hf-mirror.com/yyfz233/Pi3/resolve/main/model.safetensors?download=true'
target = root / 'model.safetensors'
# Hugging Face's blob page lists the file SHA256 separately from its Xet hash.
expected = '33580e4702ac671558aedeab1148fd08118f7ce45bdbeb99f3e3cf340062875d'
temporary = target.with_suffix('.download')
started = time.monotonic()
if not target.exists():
    with urllib.request.urlopen(url, timeout=60) as response, temporary.open('wb') as stream:
        size = int(response.headers['Content-Length'])
        count = 0
        last = started
        while block := response.read(8 * 1024 * 1024):
            stream.write(block)
            count += len(block)
            if time.monotonic() - last > 30:
                print(f'Pi3 weights: {count / size:.1%} ({count / 1e9:.2f} GB)', flush=True)
                last = time.monotonic()
    if count != size:
        raise RuntimeError('Incomplete checkpoint download')
    temporary.replace(target)
digest = hashlib.sha256()
with target.open('rb') as stream:
    while block := stream.read(8 * 1024 * 1024):
        digest.update(block)
if digest.hexdigest() != expected:
    raise RuntimeError('Published Pi3 checkpoint SHA256 mismatch')
(root / 'provenance.json').write_text(json.dumps({
    'official_url': 'https://huggingface.co/yyfz233/Pi3/resolve/main/model.safetensors',
    'download_url': url, 'sha256': digest.hexdigest(), 'bytes': target.stat().st_size,
    'verification_url': 'https://huggingface.co/yyfz233/Pi3/blob/main/model.safetensors',
    'license': 'CC BY-NC 4.0',
}, indent=2) + '\n')
print(f'Pi3 checkpoint verified: {target}', flush=True)
