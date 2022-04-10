from setuptools import setup, find_packages

setup(
    name='biotrainers',
    version='0.0.1',
    url='https://github.com/sacdallago/biotrainers.git',
    author='Christian Dallago',
    author_email='code@dallago.us',
    description='Biotrainers for embeddings',
    packages=find_packages(include=['biotrainer', 'biotrainer.*']),
    entry_points={
        'console_scripts': ['biotrainer=biotrainer.utilities.cli:main']
    }
)
