from abc import ABC

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Sequence, Callable, Union, List, Any, Iterator

from langgraph.constants import Send


class DecisionEdge(ABC):
    # The source node's name
    start_node_name: str
    # Each condition is related to one target node. The list will contain the name of the target nodes
    # that are attached to each possible condition return
    target_nodes_names: Sequence[str]

    @abstractmethod
    def _condition(self, state: Any) -> Union[str, List]:
        """
        The condition to meet to pass the edge to the target node.
        ** Important ** to note - The return string and the target_nodes_names must be the same.
        Returns:
         String  - Start node name,
          List or String - the name of the target node to pass to if the condition is mett
          Dict - Mapping
        """
        pass

    def get_decision_params(self) -> Tuple[str, Callable[..., str], Dict[str, str]]:
        return self.start_node_name, self._condition, {k: k for k in self.target_nodes_names}