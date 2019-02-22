Developer's guide
=================

Drafting new releases
---------------------

Follow these steps when drafting a new release:

1. Ensure that the master branch is passing the tests and that the `latest
   version of the documentation <https://dislib.bsc.es/en/latest>`_
   is properly being built.
2. Decide whether to issue a minor or a major release following this `guide
   <https://semver.org/>`_.

3. Update the release number accordingly in:

   - `conf.py <https://github.com/bsc-wdc/dislib/blob/master/docs/source/conf
     .py>`_.
   - `setup.py <https://github.com/bsc-wdc/dislib/blob/master/setup.py>`_.

4. Update the `changelog <https://github.com/bsc-wdc/dislib/blob/master/CHANGELOG.md>`_.

5. Draft a new release in `Github <https://github.com/bsc-wdc/
   dislib/releases>`_ using this `template <https://github
   .com/bsc-wdc/dislib/blob/master/.github/RELEASE_TEMPLATE.md>`_.

6. Create a pip package and upload it to PyPi:

   .. code:: bash

    python3 setup.py sdist bdist_wheel
    python3 -m twine upload dist/*
