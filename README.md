# introML

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Introduction to machine learning course projects.

To add the precommit hooks, run

~~~bash
pip install -r requirements-dev.txt
pre-commit install
~~~

# CPP version

To compile the cpp version you will have to type the following commands.

    mkdir build
    cd build
    cmake ..

Then you will only have to choose which project you want to compile.
For the task1a this would be

    make task1a_cpp

Now navigate to build/homeworks/task1a

There you will find yout executable.
