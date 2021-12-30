#from distutils.core import setup
from setuptools import setup,find_packages
setup(
  name = 'pymr',
  packages=find_packages("."),
  package_data = {'': ['*.dll', '*.so']},
  version = '0.0.3',
  description = 'MRI library in python',
  author = 'MRI group',
  author_email = '',
  url = '', # use the URL to the github repo
  download_url = '',
  install_requires=[
          'simpleitk','dipy','pydicom', 'nibabel', 'torch'
      ],
  keywords = ['MRI'], # arbitrary keywords
  classifiers = [],
)
