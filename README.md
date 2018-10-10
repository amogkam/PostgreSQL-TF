# PostgreSQL-TF
This repo is a fork of Postgres 9.6 source code with integration of Tensorflow C API. The configuration script was modified to link the tensorflow library when building the source. Functionality for loading a graph from a Protobuf file and calling it within the Postgres Optimizer is added.

## Installation Instructions  
1. First install Tensorflow C API following [these instructions](https://www.tensorflow.org/install/lang_c). Try the `hello_tf.c` example and check if it's able to be compiled with `gcc -ltensorflow -o hello_tf.o`. If this compilation fails, then something went wrong with the installation and/or linking.  
2. Follow [Chapter 16](https://www.postgresql.org/docs/9.6/static/installation.html) of the Postgres manual to build Postgres from source and run both the server and client. You do not need to download the tar file since the modified source code is already contained in this repo.  

## Usage
The modified source code adds functionality that loads a Tensorflow graph from a protobuf file. This Protobuf file was written from the `tf_graph_sample.py` Python file in this repo. It does a simple multiplication of 2 variables.  

The tensorflow code is located inside `src/backend/optimizer/tf/tf.c` file in the Postgres source code. This file loads the graph and runs the corresponding session. If you look inside `add_paths_to_joinrel` function inside the `src/backend/optimizer/path/joinpath.c` file, you can see that the function `tf_run` is being called. 

To get this to work however, you first need to modify the `tf_run` function inside `tf.c`. Change it so that the `file_name` is set to the absolute path of the `graph.pb` file. You can also modify the `ModelRun` function to change the inputs to the Tensorflow Session. Currently, the two inputs are set to 4.0 and 3.0, thus the output is 12.0 (4.0 * 3.0).  

Once you have made these changes, you can re-run `make install`. Then start up the server and psql, and if you run a join command in psql, you should see the tensorflow output being printed out on the server side.
