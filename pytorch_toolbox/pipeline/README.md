# Pipeline

### What is this?
The pipeline module consists of two parts, the YAML loader which parses special keywords in
the configuration file such as `!Var` and `!Ref`, and the graph constructor which
takes as input the parsed configuration file and constructs a DAG (Directed Acyclic Graph).

The reason for doing this is so that the whole training run can be defined in a configuration file, and also this forces the code to be more modular hence it will be easier for testing etc.