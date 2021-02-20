from setuptools import find_packages
from setuptools import setup

# with open('requirements.txt') as f:
#     content = f.readlines()
# requirements = [x.strip() for x in content if 'git+' not in x]

REQUIRED_PACKAGES = [
    'gcsfs==0.6.0',
    'pandas==0.24.2',
    'google-cloud-storage==1.26.0',
    'pygeohash',
    'category_encoders',
    'mlflow==1.8.0',
    'joblib==0.14.1',
    'numpy==1.18.4',
    'psutil==5.7.0',
    'pygeohash==1.2.0',
    'termcolor==1.1.0',
    'memoized-property==1.0.3',
    'scikit-learn==0.23.2'
]



setup(name='mlchartist',
      version="1.0",
      description="Project Description",
      install_requires=REQUIRED_PACKAGES,
      packages=find_packages(),
      test_suite = 'tests',
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      scripts=['scripts/mlchartist-run'],
      zip_safe=False)
