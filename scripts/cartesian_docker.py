"""Run revision experiments in the existing Docker image, never on host Python."""
import argparse
import os
from pathlib import Path
import subprocess

p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--gpu',default='0');p.add_argument('--detach',action='store_true');p.add_argument('command',nargs=argparse.REMAINDER)
a=p.parse_args()
image='gtsn-pi3:20261002'
cmd=['docker','run','--init','--network','none','--read-only','--user',f'{os.getuid()}:{os.getgid()}',
     '--gpus',f'"device={a.gpu}"','--shm-size','8g','--tmpfs','/tmp:rw,exec,size=4g',
     '--env','OMP_NUM_THREADS=1','--env','OPENBLAS_NUM_THREADS=1','--env','MKL_NUM_THREADS=1',
     '--env','LP_NUM_THREADS=1','--env','MPLCONFIGDIR=/tmp/matplotlib','--env','XDG_CACHE_HOME=/tmp/cache',
     '--env','PYTHONDONTWRITEBYTECODE=1','--env','PYTHONPATH=/workspace/src:/workspace/vendor/Pi3:/workspace/scripts',
     '--mount',f'type=bind,source={Path(__file__).resolve().parents[1]},target=/workspace,readonly',
     '--mount','type=bind,source=/run/user/1016/tsn-1k,target=/run/user/1016/tsn-1k,readonly',
     '--mount','type=bind,source=/run/user/1016/experiments,target=/run/user/1016/experiments',
     '--name',a.name]
if a.detach:cmd+=['-d']
else:cmd+=['--rm']
cmd+=[image,*a.command]
subprocess.run(cmd,check=True)
