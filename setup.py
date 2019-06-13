from setuptools import setup

setup(name='pytorch_toolbox',
      version='0.1.0',
      description='A PyTorch Toolbox',
      url='https://github.com/kevinlu1211/pytorch-toolbox',
      author='Kevin Lu',
      author_email='kevinyihchyunlu@gmail.com',
      license='MIT',
      packages=['pytorch_toolbox'],
      install_requires=[
            "tensorboardx==1.6",
            "nose==1.3.7",
            "fastprogress==0.1.15",
            "miniutils>=1.0.1",
            "pandas>=0.24.2",
            "matplotlib>=3.1.0",
      ],
      zip_safe=False)
