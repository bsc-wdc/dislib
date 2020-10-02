---
name: Bug report
about: Create a report to help us improve
title: ''
labels: bug
assignees: ''

---

**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Either provide steps to reproduce the behavior. Or attach a minimal working example.

Steps:

1. Run `dislib init` on the root of the repo
2. Issue the command `dislib exec examples/csvm-driver.py`
3. ...

Minimal example code

```
from dislib.classification import CascadeSVM
clf = CascadeSVM()
```

COMPSs launch command (where relevant)

E.g. 

```
runcompss \
    --pythonpath=$(pwd) \
    --python_interpreter=python3 \
    ./tests/__main__.py 

```

**Expected behavior**
A clear and concise description of what you expected to happen.

**Screenshots**
If applicable, add screenshots to help explain your problem.

**Environment (please complete the following information):**
 - OS: [e.g. linux mint]
 - Version [e.g. 18]
 - COMPSs version [e.g. 2.7]
 - Dislib version [e.g. 0.1.0]
 - Java / Python versions [e.g. java 8 / python 3.6.4]

**Additional context**
Add any other context about the problem here.
