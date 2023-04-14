from setuptools import setup, find_packages

setup(
    name='auto_runner',
    version='0.1',
    description='A package tries to automate the process of running a task list',
    packages=find_packages(where='src'),
    package_dir={"": "src"},
    install_requires=[
        # 'numpy',
        # 'pandas',
        'dynaconf',
        'openai',
    ],
    entry_points={
        'console_scripts': [
            # 'cchat=llminterface.interface_openai:main',
            # 'chat=llminterface.llm_wrapper:main',
            # 'config=config.config:main',
        ],
    },
)
