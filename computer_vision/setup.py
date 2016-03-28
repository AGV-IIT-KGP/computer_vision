#! /usr/bin/env python
from distutils.core import setup
setup(name='agv_cv',
      version='0.1',
      packages=['agv_cv'],
      install_requires=[
          'scipy',
          'pillow',
          'numpy',
      ],
      )
