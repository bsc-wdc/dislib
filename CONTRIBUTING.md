
Contributing to dislib
============================

How to contribute
-----------------

The preferred workflow for contributing to dislib is to fork the
[main repository](https://github.com/bsc-wdc/dislib) on
GitHub, clone, and develop on a branch. Steps:

1. Fork the [project repository](https://github.com/bsc-wdc/dislib)
   by clicking on the 'Fork' button near the top right of the page. This creates
   a copy of the code under your GitHub user account. For more details on
   how to fork a repository see [this guide](https://help.github.com/articles/fork-a-repo/).

2. Clone your fork of the dislib repo from your GitHub account to your local disk:

   ```bash
   $ git clone git@github.com:YourLogin/dislib.git
   $ cd dislib
   ```

3. Create a ``feature`` branch to hold your development changes:

   ```bash
   $ git checkout -b my-feature
   ```

   Always use a ``feature`` branch. It's good practice to never work on the ``master`` branch!

4. Develop the feature on your feature branch. Add changed files using ``git add`` and then ``git commit`` files:

   ```bash
   $ git add modified_files
   $ git commit
   ```

   to record your changes in Git, then push the changes to your GitHub account with:

   ```bash
   $ git push -u origin my-feature
   ```

5. Follow [these instructions](https://help.github.com/articles/creating-a-pull-request-from-a-fork)
to create a pull request from your fork. This will send an email to the committers.

(If any of the above seems like magic to you, please look up the
[Git documentation](https://git-scm.com/documentation) on the web, or ask a friend or another contributor for help.)

Pull Request Checklist
----------------------

We recommended that your contribution complies with the
following rules before you submit a pull request:

-  **Run the tests** before attempting to merge. You can run them locally with:
```
./run_tests.sh # it may ask your password to start the ssh daemon
```

-  **Check the code coverage**, it should at least do not decrease due to the PR. You can run them locally with:
```
pip3 install coverage
./run_coverage.sh # it may ask your password to start the ssh daemon
```

-  **Check the code style** before attempting to merge. If there is any warning the PR will be rejected. You can run them locally with:
```
pip3 install flake8
./run_style.sh
```
-  **Docker image**. All tests and code checks are run inside a docker image. If you want to run the tests in the same environment that travis will use:
```
    docker build --tag bscwdc/dislib .
    docker run -d --name dislib bscwdc/dislib
    docker exec dislib /dislib/run_ci_checks.sh
```
-  Follow the coding-guidelines defined by default [flake8](http://flake8.pycqa.org/en/latest/).

-  Give your pull request a helpful title that summarises what your
   contribution does. In some cases `Fix <ISSUE TITLE>` is enough.
   `Fix #<ISSUE NUMBER>` is not enough.

-  Often pull requests resolve one or more other issues (or pull requests).
   If merging your pull request means that some other issues/PRs should
   be closed, you should
   [use keywords to create link to them](https://github.com/blog/1506-closing-issues-via-pull-requests/)
   (e.g., `Fixes #1234`; multiple issues/PRs are allowed as long as each one
   is preceded by a keyword). Upon merging, those issues/PRs will
   automatically be closed by GitHub. If your pull request is simply related
   to some other issues/PRs, create a link to them without using the keywords
   (e.g., `See also #1234`).

-  All public methods should have informative docstrings with sample
   usage presented as doctests when appropriate.

-  Please prefix the title of your pull request `[WIP]` to indicate a work
   in progress where you expect feedback before doing more work. WIPs may be useful
   to: indicate you are working on something to avoid duplicated work,
   request broad review of functionality or API, or seek collaborators.
   WIPs often benefit from the inclusion of a
   [task list](https://github.com/blog/1375-task-lists-in-gfm-issues-pulls-comments)
   in the PR description.

-  When adding additional functionality, provide at least one
   example script in the ``examples/`` folder. Have a look at other
   examples for reference. Examples should demonstrate why the new
   functionality is useful in practice and, if possible, compare it
   to other methods available.

-  Documentation and high-coverage tests are necessary for enhancements to be
   accepted. Bug-fixes or new features should be provided with 
   [non-regression tests](https://en.wikipedia.org/wiki/Non-regression_testing).
   These tests verify the correct behavior of the fix or feature. In this
   manner, further modifications on the code base are granted to be consistent
   with the desired behavior.
   For the Bug-fixes case, at the time of the PR, this tests should fail for
   the code base in master and pass for the PR code.

Bonus points for contributions that include a performance analysis with
a benchmark script and profiling output (please report on the mailing
list or on the GitHub issue).

Filing bugs
-----------
We use GitHub issues to track all bugs and feature requests; feel free to
open an issue if you have found a bug or wish to see a feature implemented.

It is recommended to check that your issue complies with the
following rules before submitting:

-  Verify that your issue is not being currently addressed by other
   [issues](https://github.com/bsc-wdc/dislib/issues?q=)
   or [pull requests](https://github.com/bsc-wdc/dislib/pulls?q=).

-  Please ensure all code snippets and error messages are formatted in
   appropriate code blocks.
   See [Creating and highlighting code blocks](https://help.github.com/articles/creating-and-highlighting-code-blocks).

-  Please include your operating system type and version number, as well
   as your PyCOMPSs version (or docker image version)

-  Please be specific about what estimators and/or functions are involved
   and the shape of the data, as appropriate; please include a
   [reproducible](https://stackoverflow.com/help/mcve) code snippet
   or link to a [gist](https://gist.github.com). If an exception is raised,
   please provide the traceback.

New contributor tips
--------------------

A great way to start contributing to dislib is to choose a possible algorithm from sklearn or one of your interest and
get in touch with us! We are fully open to research collaborations.

Documentation
-------------

We use [numpy doc style](https://numpydoc.readthedocs.io/en/latest/format.html). We are glad to accept any sort of 
documentation: function docstrings, reStructuredText documents (like this one), tutorials, etc.
reStructuredText documents live in the source code repository under the
doc/ directory.

When you are writing documentation, it is important to keep a good
compromise between mathematical and algorithmic details, and give
intuition to the reader on what the algorithm does. It is best to always
start with a small paragraph with a hand-waving explanation of what the
method does to the data and a figure (coming from an example)
illustrating it.
