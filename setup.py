from setuptools import find_packages, setup
setup(
    name='diffgeolib',
    packages=find_packages(include=['diffgeolib']),
    version='0.1.0',
    description='Python library for differential geometry purposes. ',
    author='Isidor Schoch',
    license='MIT',
    install_requires=[],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
)
