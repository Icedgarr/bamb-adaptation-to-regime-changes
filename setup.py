try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup


SHORT = 'adaptation_to_regime_changes'


def get_requirements():
    with open('requirements.txt') as fp:
        return [x.strip() for x in fp.read().split('\n') if not x.startswith('#')]


setup(
    name='adaptation_to_regime_changes',
    packages=find_packages(),
    install_requires=get_requirements(),
    classifiers=(
        'Programming Language :: Python :: 3.6'
    ),
    description=SHORT,
    long_description=SHORT,
)
