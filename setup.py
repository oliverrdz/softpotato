from setuptools import setup, find_packages

setup(
    name='softpotato',
    version='0.0.4',
    license='GPL3',
    author='Oliver Rodriguez',
    author_email='oliver.rdz@gmail.com',
    packages=find_packages('src'),
    package_dir={'':'src'},
    url='https://github.com/oliverrdz/softpotato_py',
    keywords='Electrochemistry',
    install_requires=[
        'numpy',
    ],
)
