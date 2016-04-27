#!/usr/bin/env python

import os.path
import distutils.core
import distutils.command.build

from distutils.command.build import build
import os, sys

class BuildModelsCommand(distutils.core.Command):
    description = "Train models and save the model parameters to file"
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
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


distutils.core.setup(
    name='vessel-scoring',
    description="Tools to score fishing behviour of vessels",
    long_description='',
    packages=[
        'vessel_scoring',
    ],
    requires=[],
    version='1.0',
    author='Egil Moeller, Timothy Hochberg',
    author_email='egil@skytruth.org, tim@skytruth.org',
    url='',
    license='Apache',
    cmdclass=cmdclass
)
