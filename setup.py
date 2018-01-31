from distutils.core import setup, find_packages

config = {
    'author': 'Panagiotis Tzirakis',
    'author_email': 'panagiotis.tzirakis12@imperial.ac.uk',
    'name': 'End2You',
    'description': 'The Imperial Toolkit for Multimodal Profiling.',
    'version': '0.2.1',
    'url': 'https://github.com/end2you/end2you',
    'packages': find_packages(),
    'install_requires': ['numpy', 'moviepy', 'liac-arff'],
    'license': 'Modified BSD',
    scripts=['bin/e2u'],
    'classifiers': [
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3'
    ],
}

setup(**config)
