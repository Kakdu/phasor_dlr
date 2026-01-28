import numpy as np
from copy import deepcopy

def generate_static_measurements(
    V_r_amp, V_r_angle,
    I_r_amp, I_r_angle,
    Z, Y,
    N,
    STANDARDS_r,  # ErrorIntervals object for receiving end
    STANDARDS_s,  # ErrorIntervals object for sending end
    seed_r=1,
    seed_s=2,
):
    """
    Generate N samples of phasor measurements with measurement errors applied.

    Parameters
    ----------
    V_r_amp, I_r_amp : float
        Nominal voltage/current amplitude at receiving end
    V_r_angle, I_r_angle : float
        Nominal voltage/current angle [rad] at receiving end
    Z, Y : complex
        π-model series impedance and shunt admittance
    N : int
        Number of samples
    STANDARDS_r, STANDARDS_s : ErrorIntervals
        Error intervals for receiving and sending ends
    seed_r : int
        Random seed for receiving end
    seed_s : int
        Random seed for sending end, should not be equivalent to seed_r

    Returns
    -------
    V_r_amp_n, V_r_angle_n, I_r_amp_n, I_r_angle_n,
    V_s_amp_n, V_s_angle_n, I_s_amp_n, I_s_angle_n : np.ndarray
        Arrays of length N with measurement errors applied
    """

    # -------------------------------
    # 1) Noise-free receiving-end phasors arrays
    # -------------------------------
    V_r_amp_arr = np.full(N, V_r_amp)
    V_r_angle_arr = np.full(N, V_r_angle)
    I_r_amp_arr = np.full(N, I_r_amp)
    I_r_angle_arr = np.full(N, I_r_angle)

    # -------------------------------
    # 2) Noise-free sending-end phasors using provided pi_model function
    # -------------------------------
    from phasor_dlr.models.pi_model_dynamic import pi_model_sending_phasor
    V_s_amp_arr, V_s_angle_arr, I_s_amp_arr, I_s_angle_arr = pi_model_sending_phasor(
        V_r_amp_arr, V_r_angle_arr, I_r_amp_arr, I_r_angle_arr, Z, Y
    )

    # -------------------------------
    # 3) Apply measurement errors using previously defined functions
    # -------------------------------
    from phasor_dlr.synthetic_data.noise import apply_error_to_phasors

    # Receiving end
    V_r_amp_n, V_r_angle_n, I_r_amp_n, I_r_angle_n = apply_error_to_phasors(
        V_r_amp_arr, V_r_angle_arr, I_r_amp_arr, I_r_angle_arr,
        deepcopy(STANDARDS_r), N, seed_r
    )

    # Sending end
    V_s_amp_n, V_s_angle_n, I_s_amp_n, I_s_angle_n = apply_error_to_phasors(
        V_s_amp_arr, V_s_angle_arr, I_s_amp_arr, I_s_angle_arr,
        deepcopy(STANDARDS_s), N, seed_s
    )

    return (
        V_r_amp_n, V_r_angle_n, I_r_amp_n, I_r_angle_n,
        V_s_amp_n, V_s_angle_n, I_s_amp_n, I_s_angle_n,
        V_s_amp_arr[0], V_s_angle_arr[0], I_s_amp_arr[0], I_s_angle_arr[0],
    )



def generate_synthetic_data(
    steps,
    condParameter,
    sigma_V, sigma_I, sigma_phiV, sigma_phiI,
    conv=False,
    curr=False,
    steps_psec=100
):
    """
    Generate true and noisy measurements for UKF.

    Returns
    -------
    dict with keys:
        'T_true', 'V_r_true', 'I_r_true', 'V_s_true', 'I_s_true',
        'sent_angle_diff_true', 'received_angle_diff_true',
        'V_r_meas', 'I_r_meas', 'V_s_meas', 'I_s_meas',
        'sent_angle_meas', 'received_angle_meas', 'T_meas'
    """

    from phasor_dlr.models.phasors import phasor
    from phasor_dlr.models.pi_model_static import pi_model_sending_phasor
    from phasor_dlr.models.temperature import temperature_from_phasors


    x = np.linspace(1, steps, steps)
    
    # True receiving-end states
    T_true = 55 + 2*np.sin(np.linspace(0, 5*steps/100, steps))
    
    V_r_true = phasor(138_000, 0)
    I_r_true = phasor(
        800 + 25*np.sin(np.linspace(0, 4*steps/100, steps)),
        -np.arccos(0.8) + 0.05*np.sin(np.linspace(0, 4*steps/100, steps))
    )
    
    # Optional events
    if conv:
        startConv = steps // 4
        tauConv = steps_psec * 60 * 8
        T_true -= turn_on_exp(x, startConv, tau=tauConv, A=10)
    if curr:
        startCurr = steps // 2
        tauCurr = steps_psec * 60 * 13
        T_true += turn_on_exp(x, startCurr, tau=tauCurr, A=15)
        I_r_true.amplitude += 400 * np.concatenate([np.zeros(startCurr), np.ones(steps - startCurr)])
    
    # Sending-end phasors
    V_s_true_amp, V_s_true_angle, I_s_true_amp, I_s_true_angle = pi_model_sending_phasor(
        V_r_true.amplitude, V_r_true.angle,
        I_r_true.amplitude, I_r_true.angle,
        T_true, condParameter
    )
    
    # Angle differences
    received_angle_diff_true = V_r_true.angle - I_r_true.angle
    sent_angle_diff_true = V_s_true_angle - I_s_true_angle

    # Noisy measurements
    V_r_meas = V_r_true.amplitude * (1 + np.random.randn(steps) * sigma_V)
    I_r_meas = I_r_true.amplitude * (1 + np.random.randn(steps) * sigma_I)
    V_s_meas = V_s_true_amp * (1 + np.random.randn(steps) * sigma_V)
    I_s_meas = I_s_true_amp * (1 + np.random.randn(steps) * sigma_I)

    sent_angle_meas = sent_angle_diff_true + np.random.randn(steps) * np.sqrt(sigma_phiV**2 + sigma_phiI**2)
    received_angle_meas = received_angle_diff_true + np.random.randn(steps) * np.sqrt(sigma_phiV**2 + sigma_phiI**2)

    T_meas = np.array([
        temperature_from_phasors(
            phasor(V_r_meas[k],0.0), phasor(I_r_meas[k],sent_angle_meas[k]),
            phasor(V_s_meas[k],0.0), phasor(I_s_meas[k],received_angle_meas[k]),
            condParameter
        ) for k in range(steps)
    ])

    return dict(
        T_true=T_true,
        V_r_true=V_r_true, 
        I_r_true=I_r_true,
        V_s_true=phasor(V_s_true_amp, V_s_true_angle),
        I_s_true=phasor(I_s_true_amp, I_s_true_angle),
        sent_angle_diff_true=sent_angle_diff_true,
        received_angle_diff_true=received_angle_diff_true,
        V_r_meas=V_r_meas, 
        I_r_meas=I_r_meas,
        V_s_meas=V_s_meas, 
        I_s_meas=I_s_meas,
        sent_angle_meas=sent_angle_meas,
        received_angle_meas=received_angle_meas,
        T_meas=T_meas
    )
