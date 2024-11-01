from setuptools import setup, find_packages

setup(
    name='timbre',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'tensorflow ==  2.16.1',
        'numpy<2',  # NumPyのバージョンを1.xに固定
        'scipy',
        'librosa',
		'matplotlib',
    ],
    author='Your Name',
    author_email='your.email@example.com',
    description='A package for timbre analysis using TensorFlow',
    url='https://github.com/yourusername/timbre',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)