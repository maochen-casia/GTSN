"""Host-side standard-library launcher metadata; numerical work stays in Docker."""
import json
from pathlib import Path
import subprocess


def main():
    root = Path('/run/user/1016/experiments/gtsn_consensus_20261003')
    names = subprocess.check_output(['docker','ps','-a','--filter','name=gtsn-consensus-',
                                     '--format','{{.Names}}'], text=True).splitlines()
    records = []
    (root/'logs').mkdir(parents=True, exist_ok=True)
    for name in sorted(names):
        if not name.startswith('gtsn-consensus-'):
            continue
        item = json.loads(subprocess.check_output(['docker','inspect',name], text=True))[0]
        records.append(dict(name=name, image_id=item['Image'], command=item['Config']['Cmd'],
                            state=item['State'], mounts=item['Mounts']))
        log = subprocess.run(['docker','logs',name], stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, check=True).stdout
        (root/'logs'/f'{name}.log').write_text(log)
    (root/'container_runs.json').write_text(json.dumps(records,indent=2)+'\n')
    print(f'Archived {len(records)} project container records')


if __name__ == '__main__': main()
