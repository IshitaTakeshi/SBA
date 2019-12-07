from setuptools import setup


setup(
    name='sba',
    description='SBA',
    url='http://github.com/IshitaTakeshi/SBA',
    author='Takeshi Ishita',
    author_email='ishitah.takeshi@gmail.com',
    license='Apache 2.0',
    packages=['sparseba'],
    install_requires=['numpy'],
    tests_require=['pytest']
)
