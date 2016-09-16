from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='rlda',
      version='0.2',
      description='A module to use robust lda topics for the study of text',
      url='http://github.com/CasAndreu/rlda',
      author='Andreu Casas',
      author_email='acasas2@uw.edu',
      license='MIT',
      packages=['rlda'],
      install_requires=[
          'nltk',
          'sklearn',
          'numpy',
          'operator',
          're',
          'json',
          'string',
          'tqdm',
          'lda',
          'textmining'
      ],
      zip_safe=False,
      include_package_data=True)