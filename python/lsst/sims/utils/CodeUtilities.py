import numpy as np

def _validate_inputs(input_list, input_names, method_name):
    """
    This method will validate the inputs of other methods.

    input_list is a list of the inputs passed to a method.

    input_name is a list of the variable names associated with
    input_list

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
        raise RuntimeError("The arg %s imput to method %s " % (input_names[0], method_name)
                           + "should be either a float or a numpy array")

    valid_type = True
    bad_names = []
    for ii, nn in zip(input_list, input_names):
        if not isinstance(ii, desired_type):
            valid_type = False
            bad_names.append(nn)

    if not valid_type:
        msg = "The input arguments:\n"
        for nn in bad_names:
            msg += "%s,\n" % nn
        msg += "passed to %s " % method_name
        msg += "need to be either floats or numpy arrays\n"
        msg += "and the same type as the argument %s" % input_names[0]
        raise RuntimeError(msg)

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
