from setuptools import setup, find_packages

setup(
    name='controllab',
    version='0.1.0',
    # find_packages agora procura a partir de 'src'
    packages=find_packages(where='src'),
    # Diz ao setuptools que os pacotes estão sob o diretório 'src'
    package_dir={'': 'src'},
    description='Uma biblioteca de engenharia de controlo simbólica e pedagógica.',
    # Adicione aqui as dependências de 'requirements.txt' no futuro
    install_requires=[
        'numpy',
        'scipy',
        'sympy',
        'control',
        'matplotlib',
        'flask' # ou fastapi
    ],
)
