try:
  from distutils.core import setup, find_packages
except:
  from setuptools import setup, find_packages

requirements = [
    'numPy >= 1.19.2',
    'torch >= 1.7',
    'torchvision >= 0.8.1',
    'moviePy >= 1.0.3',
    'scikit-learn >= 0.23.2',
    'h5py >= 2.10.0',
    'facenet-pytorch >= 2.5',
    'tqdm >= 4.51.0'
]

config = {
    'author': 'Panagiotis Tzirakis',
    'author_email': 'panagiotis.tzirakis12@imperial.ac.uk',
    'name': 'End2You',
    'description': 'The Imperial Toolkit for Multimodal Profiling.',
    'version': '1.0',
    'url': 'https://github.com/end2you/end2you',
    'packages': find_packages(),
    'install_requires': requirements,
    'license': 'Modified BSD',
    'entry_points':{
        "console_scripts": [
            "e2u = end2you.main:main"
        ]
    },
    'include_package_data':True,
    'classifiers': [
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3.7'
    ],
    'url':'https://github.com/end2you/end2you',
    'keywords':'multimodal-deep-learning machine-learning end-to-end-learning science research'
}

setup(**config)
