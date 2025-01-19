from setuptools import setup, find_packages

classifiers = [
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Education',
    'Operating System :: Microsoft :: Windows :: Windows 10',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3'
]

setup(
    name='polynomial-fit',
    version='0.0.1',
    description='Fit a polynomial of any order and dimension to a dataset',
    long_description=open('README.md').read() + '\n\n' + open('CHANGELOG.txt').read(),
    url='',
    author='Seth Reed',
    author_email='seth.reed01@gmail.com',
    license='MIT',
    classifiers=classifiers,
    keywords='polynomial',
    packages=find_packages(),
    install_requires=['']
)