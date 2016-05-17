import numpy as np

def _validate_inputs(input_list, method_name):
    """
    This method will validate the inputs of other methods.

    input_list is a list of the inputs passed to a method.

    method_name is the name of the method whose input is being validated.

    _validate_inputs will verify that all of the inputs in input_list are:

    1) of the same type
    2) either floats or numpy arrays
    3) if they are numpy arrays, they all have the same length

    If any of these criteria are violated, a RuntimeError will be raised

    returns True if the inputs are numpy arrays; False if not
    """

    if isinstance(input_list[0], np.ndarray):
        desired_type = np.ndarray
    elif isinstance(input_list[0], np.float):
        desired_type = np.float
    else:
        raise RuntimeError("The inputs to %s should all be the same type," % method_name
                           + " either floats or numpy arrays")

    valid_type = True
    for ii in input_list:
        if not isinstance(ii, desired_type):
            valid_type = False

    if not valid_type:
        raise RuntimeError("The inputs to %s should all be the same type," % method_name
                           + " either floats or numpy arrays")

    if desired_type is np.ndarray:
        same_length=True
        for ii in input_list:
            if len(ii) != len(input_list[0]):
                same_length = False
        if not same_length:
            raise RuntimeError("The arrays input to %s " % method_name
                               + "all need to have the same length")

    if desired_type is np.ndarray:
        return True

    return False
