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

4. Update the `change log <https://github.com/bsc-wdc/dislib/blob/master/CHANGELOG.md>`_.

5. Create and tag a docker image for the release running the following at the repo's root (change ``VERSION`` accordingly):
   
   - Create the image:
     
     .. code:: bash   
     
      docker build -t bscwdc/dislib:VERSION
   
   - Log in and push it to dockerhub
   
     .. code:: bash
     
      docker login -u DOCKERHUB_USER -p DOCKERHUB_PASSWORD
      docker push bscwdc/dislib:VERSION

6. Update the version number in `dislib_cmd.py <https://github.com/bsc-wdc/dislib/blob/master/bin/dislib_cmd.py>`_ where it says:

   .. code:: python

    image_name = 'bscwdc/dislib:VERSION'   

7. Draft a new release in `Github <https://github.com/bsc-wdc/
   dislib/releases>`_ using this `template <https://github
   .com/bsc-wdc/dislib/blob/master/.github/RELEASE_TEMPLATE.md>`_.

8. Create a pip package and upload it to PyPi:

   .. code:: bash

    python3 setup.py sdist bdist_wheel
    python3 -m twine upload dist/*
