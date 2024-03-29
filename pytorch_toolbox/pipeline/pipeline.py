import logging
import functools
import networkx as nx

from .yaml_loader import load_config_from_string, load_config_from_path, Reference
from .graph_construction import (
    flatten_resources_dict,
    find_references,
    replace_arguments,
    replace_should_run,
)


class Pipeline:
    def __init__(self, graph, config, state_dict={}):
        self.graph = graph
        self.config = config
        self.state_dict = {**state_dict}

    @classmethod
    def create_from_config(cls, config, lookup, state_dict={}):
        assert (
            config.get("Resources") is not None
        ), "There is no Resources key in the configuration file"
        graph = nx.DiGraph()
        flattened_resources = flatten_resources_dict(config["Resources"])
        graph = cls._add_nodes_to_graph(graph, flattened_resources, lookup)
        graph = cls._add_edges_to_graph(graph)
        return cls(graph, config, state_dict)

    @classmethod
    def create_from_config_string(
        cls, config_string, lookup, with_variable_replacement=True
    ):
        raw_config = load_config_from_string(config_string)
        replaced_config = load_config_from_string(
            config_string, with_variable_replacement=with_variable_replacement
        )
        return cls.create_from_config(
            replaced_config, lookup, state_dict=dict(raw_config=raw_config)
        )

    @classmethod
    def create_from_config_path(cls, config_path, lookup, with_variable_replacement=True):
        raw_config = load_config_from_path(config_path)
        replaced_config = load_config_from_path(
            config_path, with_variable_replacement=with_variable_replacement
        )
        return cls.create_from_config(
            replaced_config, lookup, state_dict=dict(raw_config=raw_config)
        )

    @staticmethod
    def _add_nodes_to_graph(graph, resources, lookup):
        logging.info("Adding nodes to graph")
        for name, resource in resources.items():
            assert (
                "properties" in resource
            ), f"The properties key isn't defined for node: {name}"
            references = find_references(resource)
            properties = load_properties_with_default_values(resource["properties"], lookup)
            node = Node(name=name, references=references, **properties)
            graph.add_node(name, node=node)
        return graph

    @staticmethod
    def _add_edges_to_graph(graph):
        logging.info("Adding edges to graph")
        for name, node_wrapper in graph.nodes(data=True):
            node = node_wrapper["node"]
            for reference in node.references:
                referenced_node_name = reference.ref_node_name

                assert (
                    referenced_node_name in graph.nodes
                ), f"The reference: {referenced_node_name} in node: {node.name} does not exist"

                referenced_node = graph.nodes(data=True)[referenced_node_name]["node"]
                referenced_node.referenced_by.append(node)

                graph.add_edge(referenced_node_name, name)
        return graph

    @property
    def sorted_node_names(self):
        return list(nx.algorithms.dag.topological_sort(self.graph))

    def get_node(self, name):
        return self.graph.nodes(data=True)[name]["node"]

    def get_node_output_lookup(self, name):
        return self.get_node(name).output

    def get_node_output(self, name, output_lookup_key=None):
        output_lookup = self.get_node_output_lookup(name)
        if output_lookup is None:
            return None

        if output_lookup_key is None and len(output_lookup) == 1:
            output = list(output_lookup.values())[0]
            return output
        else:
            return output_lookup[output_lookup_key]

    def run(self, to_node=None):
        logging.info(self.sorted_node_names)
        logging.info("Running nodes in graph")
        for node_name in nx.algorithms.dag.topological_sort(self.graph):
            logging.debug(f"Processing node: {node_name}")
            node = self.graph.nodes(data=True)[node_name]["node"]
            logging.debug(f"Updating should_run property for node: {node_name}")
            self._update_node_should_run(node)
            if node.should_run:
                self._run_node(node)

            if to_node == node_name:
                break

    def _run_node(self, node):
        self._replace_node_argument_references(node)
        logging.debug(f"Running node: {node.name}")
        node.create_output()

    def _update_node_should_run(self, current_node):
        current_node_should_run = self._get_node_should_run_property(current_node)
        current_node.should_run = current_node_should_run

        for referenced_node in current_node.referenced_by:
            if (
                isinstance(referenced_node.should_run, Reference)
                or referenced_node.should_run is False
            ):
                continue
            else:
                referenced_node.should_run = current_node_should_run
                self._update_node_should_run(referenced_node)

    def _get_node_should_run_property(self, node):
        if isinstance(node.should_run, Reference):
            node_should_run = replace_should_run(self.graph, node)
        else:
            node_should_run = node.should_run
        return node_should_run

    def _replace_node_argument_references(self, node):
        logging.debug(f"Replacing references for node: {node.name}")
        logging.debug(f"Original arguments: {node.arguments}")
        reference_replaced_arguments = replace_arguments(self.graph, self.state_dict, node)
        node.reference_replaced_arguments.update(**reference_replaced_arguments)
        logging.debug(f"Replaced arguments: {node.reference_replaced_arguments}")


class Node:
    def __init__(
        self, name, references, pointer, partial, arguments, output_names, should_run
    ):
        self.name = name
        self.references = references
        self.referenced_by = []
        self.pointer = pointer
        self.partial = partial
        self.arguments = arguments
        self.output_names = output_names
        self.should_run = should_run
        self.reference_replaced_arguments = {}
        self.output = None

    def __repr__(self):
        return f"""
            Name: {self.name}
            Pointer: {self.pointer}
            Arguments: {self.arguments}
            Should Run: {self.should_run}
        """

    def create_output(self):
        assert not isinstance(
            self.output_names, str
        ), f"The output_names for node: {self.name} can't be a string, only a list"

        self._call_pointer_and_set_output()

    def _call_pointer_and_set_output(self):
        if self.partial:
            assert (
                len(self.output_names) == 1
            ), f"If the output of node: {self.name} is partial, then there should be one output, {len(self.output_names)} outputs are found"
            self.output = {
                self.output_names[0]: functools.partial(
                    self.pointer, **self.reference_replaced_arguments
                )
            }
        else:
            try:
                output = self.pointer(**self.reference_replaced_arguments)
            except TypeError as e:
                logging.error(e)
                logging.error(f"This error was produced when calling node:\n{self}")
                logging.error(e, exc_info=True)
                exit(1)
            if output is not None:
                assert (
                    len(self.output_names) > 0
                ), f"There are no output_names defined in properties for the node: {self.name}"
                iterable_output = [output] if len(self.output_names) == 1 else output
                self.output = {
                    output_name: output_value
                    for output_name, output_value in zip(self.output_names, iterable_output)
                }


def load_properties_with_default_values(properties, lookups):
    pointer_name = properties["pointer"]
    assert pointer_name in lookups, f"There is no lookup called: {pointer_name}"
    return {
        "pointer": lookups[pointer_name],
        "partial": properties.get("partial", False),
        "arguments": properties.get("arguments", {}),
        "output_names": properties.get("output_names", []),
        "should_run": properties.get("should_run", True),
    }
