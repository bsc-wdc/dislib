import os

import docker
from docker.types import Mount

client = docker.from_env()
master_name = 'dislib-master'


def _is_running(name: str):
    cs = client.containers.list(filters={'name': name})

    return len(cs) > 0


def _start_daemon(working_dir: str = "", restart: bool = False):
    containers = client.containers.list(filters={'name': master_name},
                                        all=True)
    assert len(containers) < 2  # never should we run 2 masters

    if restart:
        for c in containers:
            c.remove(force=True)

    if not _is_running(master_name):
        if not working_dir:
            working_dir = os.getcwd()

        print("Staring %s container in dir %s" % (master_name, working_dir))

        # mount target dir needs to be absolute
        target_dir = '/home/user'

        user_dir = Mount(target=target_dir,
                         source=working_dir,
                         type='bind')

        compss_log_dir = Mount(target='/root/.COMPSs',
                               source=os.environ['HOME'] + '/.COMPSs',
                               type='bind')

        mounts = [user_dir, compss_log_dir]

        client.containers.run(image='dislib', name=master_name,
                              mounts=mounts, detach=True)


def _stop_daemon():
    containers = client.containers.list(filters={'name': master_name})
    for c in containers:
        # autoremove enabled so stop is enough, no need to call remove()
        c.remove(force=True)


def _get_master():
    master = client.containers.list(filters={'name': master_name})[0]

    return master


def _exec_in_daemon(cmd: str):
    print("Executing cmd: %s" % cmd)
    if not _is_running(master_name):
        _start_daemon()

    master = _get_master()
    exit_code, output = master.exec_run(cmd, workdir='/home/user')

    print("Exit code: %s" % exit_code)
    for line in [l for l in output.decode().split('\n')]:
        print(line)
