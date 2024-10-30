import pytest
import numpy as np
from anf_generator import generate_signals, CoherenceMatrix, MixingMatrix

@pytest.fixture
def setup_params():
    num_channels = 4
    params = CoherenceMatrix.Parameters(
        mic_positions=np.array([[0.04 * i, 0, 0] for i in range(num_channels)]),
        sc_type="spherical",
        sample_frequency=16000,
        nfft=1024,
    )
    return params

def test_generate_signals_standard(setup_params):
    input_signals = np.random.randn(4, 16000)
    output_signals, coherence_target, mixing_matrix = generate_signals(
        input_signals, setup_params, decomposition='evd', processing='standard'
    )
    assert output_signals.shape == input_signals.shape
    assert isinstance(coherence_target, CoherenceMatrix.CoherenceMatrix)
    assert isinstance(mixing_matrix, MixingMatrix.MixingMatrix)

def test_generate_signals_balance_smooth(setup_params):
    input_signals = np.random.randn(4, 16000)
    output_signals, coherence_target, mixing_matrix = generate_signals(
        input_signals, setup_params, decomposition='evd', processing='balance+smooth'
    )
    assert output_signals.shape == input_signals.shape
    assert isinstance(coherence_target, CoherenceMatrix.CoherenceMatrix)
    assert isinstance(mixing_matrix, MixingMatrix.MixingMatrix)

def test_generate_signals_invalid_decomposition(setup_params):
    input_signals = np.random.randn(4, 16000)
    with pytest.raises(ValueError):
        generate_signals(input_signals, setup_params, decomposition='invalid', processing='standard')

def test_generate_signals_invalid_processing(setup_params):
    input_signals = np.random.randn(4, 16000)
    with pytest.raises(ValueError):
        generate_signals(input_signals, setup_params, decomposition='evd', processing='invalid')

def test_generate_signals_corcos(setup_params):
    setup_params.sc_type = 'corcos'
    setup_params.speed = 20
    setup_params.direction = 60
    input_signals = np.random.randn(4, 16000)
    output_signals, coherence_target, mixing_matrix = generate_signals(
        input_signals, setup_params, decomposition='evd', processing='standard'
    )
    assert output_signals.shape == input_signals.shape
    assert isinstance(coherence_target, CoherenceMatrix.CoherenceMatrix)
    assert isinstance(mixing_matrix, MixingMatrix.MixingMatrix)

def test_generate_signals_cylindrical(setup_params):
    setup_params.sc_type = 'cylindrical'
    input_signals = np.random.randn(4, 16000)
    output_signals, coherence_target, mixing_matrix = generate_signals(
        input_signals, setup_params, decomposition='evd', processing='standard'
    )
    assert output_signals.shape == input_signals.shape
    assert isinstance(coherence_target, CoherenceMatrix.CoherenceMatrix)
    assert isinstance(mixing_matrix, MixingMatrix.MixingMatrix)

def test_generate_signals_chd(setup_params):
    input_signals = np.random.randn(4, 16000)
    output_signals, coherence_target, mixing_matrix = generate_signals(
        input_signals, setup_params, decomposition='chd', processing='standard'
    )
    assert output_signals.shape == input_signals.shape
    assert isinstance(coherence_target, CoherenceMatrix.CoherenceMatrix)
    assert isinstance(mixing_matrix, MixingMatrix.MixingMatrix)

def test_generate_signals_balance(setup_params):
    input_signals = np.random.randn(4, 16000)
    output_signals, coherence_target, mixing_matrix = generate_signals(
        input_signals, setup_params, decomposition='evd', processing='balance'
    )
    assert output_signals.shape == input_signals.shape
    assert isinstance(coherence_target, CoherenceMatrix.CoherenceMatrix)
    assert isinstance(mixing_matrix, MixingMatrix.MixingMatrix)

def test_generate_signals_smooth(setup_params):
    input_signals = np.random.randn(4, 16000)
    output_signals, coherence_target, mixing_matrix = generate_signals(
        input_signals, setup_params, decomposition='evd', processing='smooth'
    )
    assert output_signals.shape == input_signals.shape
    assert isinstance(coherence_target, CoherenceMatrix.CoherenceMatrix)
    assert isinstance(mixing_matrix, MixingMatrix.MixingMatrix)
