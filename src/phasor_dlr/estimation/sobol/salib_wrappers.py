from SALib.analyze import sobol
import numpy as np

def run_sobol_analysis(problem, Y, calc_second_order=True):
    """
    Wrapper around SALib's sobol.analyze that clips negative artefacts.
    """
    Si = sobol.analyze(problem, Y, print_to_console=False, calc_second_order=calc_second_order)

    # Clip small negative numerical artefacts
    for key in ["S1","ST","S2","S1_conf","ST_conf"]:
        Si[key] = np.clip(Si[key], 0, None)

    return Si