import setuptools
from os import path

VERSION = '0.2.13'

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), 'r') as f:
    long_description = f.read()

setuptools.setup(name='vision-evaluation',
                 author='Ping Jin, Shohei Ono',
                 description="Evaluation metric codes for various vision tasks.",
                 long_description=long_description,
                 long_description_content_type="text/markdown",
                 url='https://github.com/microsoft/vision-evaluation',
                 version=VERSION,
                 license='MIT',
                 python_requires='>=3.7',
                 packages=setuptools.find_packages(),
                 keywords='vision metric evaluation classification detection',
                 classifiers=[
                     'Development Status :: 4 - Beta',
                     'Intended Audience :: Developers',
                     'License :: OSI Approved :: MIT License',
                     'Programming Language :: Python :: 3.7',
                     'Programming Language :: Python :: 3.8',
                     'Programming Language :: Python :: 3.9',
                     'Programming Language :: Python :: 3.10',
                 ],
                 install_requires=[
                     'numpy',
                     'scikit-learn>=1.0',
                     'pycocotools',
                     'pycocoevalcap',
                     'opencv-python-headless',
                     'Pillow>=6.2.2'
                 ])
