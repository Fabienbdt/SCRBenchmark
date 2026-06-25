from setuptools import setup

about = {}
with open('torchlars/__version__.py') as f:
    exec(f.read(), about)
version = about['__version__']
del about

with open('README.md') as f:
    long_description = f.read()

setup(
    name='torchlars',
    version=version,
    author='Kakao Brain (vendored by SCRBenchmark)',
    description='LARS optimizer – pure Python build (no CUDA ext)',
    long_description=long_description,
    long_description_content_type='text/markdown',
    zip_safe=False,
    packages=['torchlars'],
    install_requires=['torch'],
    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
    ],
)
