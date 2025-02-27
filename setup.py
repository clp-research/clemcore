from setuptools import setup, find_packages

setup(
    name='clembench',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        "pyyaml>=6.0",  # Logging
        "numpy>=1.24.3",  # Score computation
        "retry>=0.9.2",  # API call utility
        "tqdm>=4.65.0",
        "nltk>=3.8.1"
    ],  # tbd
    extras_require={
        "backends": [
            "aleph-alpha-client==7.0.1",
            "openai==1.12.0",
            "anthropic==0.16.0",
            "cohere==4.48",
            "google-generativeai==0.5.3",
            "mistralai==0.0.12"
        ],
        "huggingface": [
            "torch==2.1.1",  # fix pytorch version
            "transformers==4.43.1",  # Huggingface
            "sentencepiece==0.1.99",  # FLAN model
            "accelerate==0.25.0",  # FLAN model
            "einops==0.6.1",  # FALCON model
            "protobuf==4.21.6",
            "bitsandbytes==0.39.0"
        ],
        "eval": [
            "scikit-learn==1.2.2",
            "matplotlib==3.7.1",
            "pandas==2.0.1",
            "seaborn==0.12.2",
            "jupyter==1.0.0",
        ],
        "slurk": [
            "python-engineio==4.4.0",
            "python-socketio==5.7.2",
            "websocket-client"
        ]
    },
    entry_points={
        'console_scripts': [
            'clem=clemcore.cli:main',
        ],
    },
    include_package_data=True,
    package_data={
        "clembench": [
            "clemcore/utils/logging.yaml",
            "clemcore/utils/chat-two-tracks.css",
            "clemcore/backends/model_registry.json"
        ]
    },
    description="The cLLM (chat-optimized Large Language Model, 'clem') framework tests such models' ability "
                "to engage in games, that is, rule-constituted activities played using language.",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Philipp Sadler',
    author_email='first.last@uni-potsdam.de',
    url='https://github.com/clp-research/clembench',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)
