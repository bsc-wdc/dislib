#!/usr/bin/python
import os
import shutil
import subprocess
 
def main():
    p = subprocess.Popen(['ps', '-ef'], stdout=subprocess.PIPE)
    killed_count = -1
    for line in p.stdout.readlines():
        if 'compss' in line.decode() or 'COMPSs' in line.decode():
            candidates = line.decode().split(" ")[1:]
            for cand in candidates:
                if cand:
                    pid = cand
                    break
            subprocess.Popen(['kill', '-9', pid])
            killed_count += 1
    print('%d total processes killed'%killed_count)
 
 
if __name__ == "__main__":
    main()
