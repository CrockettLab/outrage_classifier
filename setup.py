import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(name='outrageclf',
      version='0.1.5',
      description='Outrage Classifier - developed by the Crockett Lab',
      long_description=README,
      url='https://github.com/CrockettLab/outrage_classifier',
      install_requires=[
            'emoji',
            'joblib',
            'keras',
            'nltk',
            'numpy',
            'sklearn',
            'tensorflow'
      ],
      author='Tuan Nguyen Doan',
      author_email='tuan.nguyen.doan@aya.yale.edu',
      license='GNU General Public License v3.0',
      packages=['outrageclf'],
      include_package_data=True,
      zip_safe=False)
