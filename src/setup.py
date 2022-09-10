from setuptools import setup, find_packages, Extension

setup(
    name='finalmodule',
    version='0.0.1',
    author="mohammad daghash and ram elgov",
    install_requires=['invoke'],
    packages=find_packages(),
    license='GPL-2',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: Implementation :: CPython'
    ],
    ext_modules=[
        Extension(
            'finalmodule',
            ['spkmeansmodule.c', 'spkmeans.c']
        )
    ]
)




