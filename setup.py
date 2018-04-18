import setuptools

if __name__ == "__main__":

    setuptools.setup(

        name='ssanchors',

        version="0.1.0",

        description=(
            'Generate anchor stimuli for MUSHRA-style experiments for '
            'assessing source separation algorithms'),

        url='https://github.com/cvssp/source-separation-anchors',

        # Your contact information
        author='Dominic Ward; Hagen Wierstorf',
        author_email='contactdominicward+github@gmail.com',

        # License
        license='MIT',

        # Packages in this project
        # find_packages() finds all these automatically for you
        packages=setuptools.find_packages(),

        entry_points={
            'console_scripts': [
                'ssanchors=ssanchors.cli:ssanchors',
            ],
        },

        # Dependencies, this installs the entire Python scientific
        # computations stack
        install_requires=[
            'numpy',
            'untwist',
        ],

        classifiers=[
            'Development Status :: 4 - Beta',
            'Environment :: Console',
            'Environment :: Plugins',
            'Intended Audience :: Telecommunications Industry',
            'Intended Audience :: Science/Research',
            'Programming Language :: Python :: 3.6',
            'Topic :: Multimedia :: Sound/Audio :: Analysis',
            'Topic :: Multimedia :: Sound/Audio :: Sound Synthesis'
        ],

        zip_safe=False,
    )
