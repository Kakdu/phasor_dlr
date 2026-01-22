import numpy as np
import pytensor
import pytensor.tensor as pt

import numpy as np

def pi_model_sending_phasor(
    V_r_amp, V_r_angle,
    I_r_amp, I_r_angle,
    Z, Y
):
    """
    Compute sending-end phasors (V_s, I_s) from receiving-end phasors (V_r, I_r)
    using a π-model transmission line.

    Supports scalar or ndarray inputs for V, I, Z, and Y via NumPy broadcasting.

    Parameters
    ----------
    V_r_amp : float or np.ndarray
        Receiving-end voltage amplitude(s)
    V_r_angle : float or np.ndarray
        Receiving-end voltage angle(s) [rad]
    I_r_amp : float or np.ndarray
        Receiving-end current amplitude(s)
    I_r_angle : float or np.ndarray
        Receiving-end current angle(s) [rad]
    Z : complex or np.ndarray
        Series impedance(s) (Ohms)
    Y : complex or np.ndarray
        Total shunt admittance(s) (Siemens)

    Returns
    -------
    V_s_amp : np.ndarray
        Sending-end voltage amplitude(s)
    V_s_angle : np.ndarray
        Sending-end voltage angle(s) [rad]
    I_s_amp : np.ndarray
        Sending-end current amplitude(s)
    I_s_angle : np.ndarray
        Sending-end current angle(s) [rad]
    """

    # Convert all inputs to arrays (broadcast-safe)
    V_r_amp   = np.asarray(V_r_amp, dtype=np.float64)
    V_r_angle = np.asarray(V_r_angle, dtype=np.float64)
    I_r_amp   = np.asarray(I_r_amp, dtype=np.float64)
    I_r_angle = np.asarray(I_r_angle, dtype=np.float64)

    Z = np.asarray(Z, dtype=np.complex128)
    Y = np.asarray(Y, dtype=np.complex128)

    # Convert to complex phasors
    V_r = V_r_amp * np.exp(1j * V_r_angle)
    I_r = I_r_amp * np.exp(1j * I_r_angle)

    # π-model calculations (broadcasted)
    I_shunt_r = (Y / 2) * V_r
    I_series = I_r + I_shunt_r

    V_s = V_r + I_series * Z

    I_shunt_s = (Y / 2) * V_s
    I_s = I_series + I_shunt_s

    # Convert back to magnitude / angle
    V_s_amp = np.abs(V_s)
    V_s_angle = np.angle(V_s)
    I_s_amp = np.abs(I_s)
    I_s_angle = np.angle(I_s)

    return V_s_amp, V_s_angle, I_s_amp, I_s_angle



def pi_model_sending_phasor_pytensor(
    V_r_amp, V_r_angle,
    I_r_amp, I_r_angle,
    Z_re, Z_im,
    Y_re, Y_im
):
    """
    PyMC / PyTensor compatible π-model transmission line.

    All inputs must be PyTensor variables or compatible constants.

    Parameters
    ----------
    V_r_amp, V_r_angle : pt.Tensor
        Receiving-end voltage amplitude and angle [rad]
    I_r_amp, I_r_angle : pt.Tensor
        Receiving-end current amplitude and angle [rad]
    Z_re, Z_im : float or pt.Tensor
        Series impedance real and imaginary parts
    Y_re, Y_im : float or pt.Tensor
        Total shunt admittance real and imaginary parts

    Returns
    -------
    V_s_amp, V_s_angle, I_s_amp, I_s_angle : pt.Tensor
    """

    # --- Receiving-end phasors (rectangular) ---
    V_r_re = V_r_amp * pt.cos(V_r_angle)
    V_r_im = V_r_amp * pt.sin(V_r_angle)

    I_r_re = I_r_amp * pt.cos(I_r_angle)
    I_r_im = I_r_amp * pt.sin(I_r_angle)

    # --- Shunt current at receiving end: (Y/2) * V_r ---
    Y2_re = Y_re / 2
    Y2_im = Y_im / 2

    I_shunt_r_re = V_r_re * Y2_re - V_r_im * Y2_im
    I_shunt_r_im = V_r_re * Y2_im + V_r_im * Y2_re

    # --- Series current ---
    I_series_re = I_r_re + I_shunt_r_re
    I_series_im = I_r_im + I_shunt_r_im

    # --- Voltage drop across series impedance ---
    dV_re = I_series_re * Z_re - I_series_im * Z_im
    dV_im = I_series_re * Z_im + I_series_im * Z_re

    V_s_re = V_r_re + dV_re
    V_s_im = V_r_im + dV_im

    # --- Shunt current at sending end ---
    I_shunt_s_re = V_s_re * Y2_re - V_s_im * Y2_im
    I_shunt_s_im = V_s_re * Y2_im + V_s_im * Y2_re

    # --- Total sending-end current ---
    I_s_re = I_series_re + I_shunt_s_re
    I_s_im = I_series_im + I_shunt_s_im

    # --- Convert back to amplitude and angle ---
    V_s_amp = pt.sqrt(V_s_re**2 + V_s_im**2)
    V_s_angle = pt.arctan2(V_s_im, V_s_re)

    I_s_amp = pt.sqrt(I_s_re**2 + I_s_im**2)
    I_s_angle = pt.arctan2(I_s_im, I_s_re)

    return V_s_amp, V_s_angle, I_s_amp, I_s_angle