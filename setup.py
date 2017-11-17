from setuptools import setup
from setuptools import find_packages


setup(
    name='copy_seq2seq',
    version='0.0.1',
    description='Copy Seq2Seq model',
    author='Igor Shalyminov',
    author_email='ishalyminov@gmail.com',
    url='https://github.com/ishalyminov/copy_seq2seq',
    download_url='https://github.com/ishalyminov/copy_seq2seq.git',
    license='MIT',
    install_requires=['tensorflow-gpu==1.4.0'],
    packages=find_packages(),
    package_data={'copy_seq2seq': ['*']}
)
