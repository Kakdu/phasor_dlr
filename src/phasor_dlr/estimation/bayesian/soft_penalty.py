import pymc as pm
import pytensor.tensor as pt

def soft_temperature_bounds(T, lower=30.0, upper=70.0, scale=5.0, name="T_soft_bounds", enabled=True):
    """
    Adds a soft quadratic penalty to T if it goes outside [lower, upper].
    
    Parameters
    ----------
    T : PyMC random variable
        Latent temperature variable
    lower : float
        Lower bound for soft penalty
    upper : float
        Upper bound for soft penalty
    scale : float
        Scaling factor for penalty (controls steepness)
    name : str
        Name of the PyMC Potential
    enabled : bool
        If False, does nothing
    """
    if not enabled:
        return  # No penalty applied
    
    # Quadratic penalties outside bounds
    penalty = pt.switch(
        T < lower, -0.5 * ((lower - T) / scale) ** 2,
        pt.switch(T > upper, -0.5 * ((T - upper) / scale) ** 2, 0.0)
    )
    
    pm.Potential(name, penalty)
