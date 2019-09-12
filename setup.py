#!/usr/bin/env python

import os.path
from setuptools import find_packages
from setuptools import setup
import os.path
import sys

DEPENDENCIES = [
    "numpy",
    "scikit-learn==0.20.0",
    "scipy",
    "rolling_measures"
]


class BuildModelsCommand(distutils.core.Command):
    description = "Train models and save the model parameters to file"
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        if os.path.exists(os.path.join(os.path.dirname(__file__), 'vessel_scoring', 'models')):
            return
        print "Training models...",
        sys.stdout.flush()
        import vessel_scoring.models
        vessel_scoring.models.train_models()
        print "done."
        sys.stdout.flush()

class build(distutils.command.build.build):
    def run(self):
        self.run_command("build_models")
        distutils.command.build.build.run(self)

cmdclass = {}
cmdclass['build_models'] = BuildModelsCommand
cmdclass['build'] = build


# distutils.core.setup(
#     name='vessel-scoring',
#     description="Tools to score fishing behavior of vessels",
#     long_description='',
#     packages=[
#         'vessel_scoring',
#     ],
#     # package_data={
#     #     'vessel_scoring': ['models/*']},
#     install_requires=DEPENDENCIES,
#     # extras_require={
#     #     'dev': ['matplotlib', 'ipython', 'coveralls']},
#     version='1.01',
#     # author='Egil Moeller, Timothy Hochberg',
#     # author_email='egil@skytruth.org, tim@skytruth.org',
#     # url='',
#     # license='Apache',
# #    cmdclass=cmdclass
# )

setup(
    name='vessel-scoring',
    version='1.0.1',
    description='Tools to score fishing behavior of vessels',
    author="Global Fishing Watch",
    author_email="info@globalfishingwatch.org",
    license="Apache 2",
    packages=find_packages(),
    install_requires=DEPENDENCIES 
)
