from setuptools import setup

setup(name='starspot',
      version='0.1',
      description='Tools for measuring stellar rotation periods',
      url='http://github.com/RuthAngus/starspot',
      download_url = "https://github.com/RuthAngus/starspot/archive/v0.1.tar.gz",
      author='Ruth Angus',
      author_email='ruthangus@gmail.com',
      license='MIT',
      packages=['starspot'],
      install_requires=['numpy', 'pandas', 'h5py', 'tqdm', 'emcee', 'exoplanet', 'pymc3', 'theano', 'astropy', 'matplotlib', 'scipy', 'kplr'],
      zip_safe=False)
