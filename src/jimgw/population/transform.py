from typing import Tuple, List, Dict
from jimgw.transforms import NtoNTransform, Float


class NullTransform(NtoNTransform):
    """
    Null transformation that does nothing to the input data.
    """

    def __init__(self, name_mapping: Tuple[List[str], List[str]]):
        super().__init__(name_mapping)

        # Ensure that the input and output name mappings are the same length
        if len(name_mapping[0]) != len(name_mapping[1]):
            raise ValueError("Input and output name mappings must have the same length.")

        # The transform function simply returns the input as-is
        def null_transform(x: Dict[str, Float]) -> Dict[str, Float]:
            return {key: x[key] for key in name_mapping[0]}

        self.transform_func = null_transform