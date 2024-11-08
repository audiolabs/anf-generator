import pytest
import numpy as np
from anf_generator import estimate_coherence, generate_signals, CoherenceMatrix, MixingMatrix

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

@pytest.fixture
def setup_params_corcos():
    num_channels = 4
    params = CoherenceMatrix.Parameters(
        mic_positions=np.array([[0.04 * i, 0, 0] for i in range(num_channels)]),
        sc_type="corcos",
        sample_frequency=16000,
        nfft=1024,
        speed = 20,
        direction = 30,
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

def test_generate_signals_debug_processing(setup_params):
    input_signals = np.random.randn(4, 16000)
    output_signals, coherence_target, mixing_matrix = generate_signals(
        input_signals, setup_params, decomposition='evd', processing='debug'
    )
    assert output_signals.shape == input_signals.shape
    assert isinstance(coherence_target, CoherenceMatrix.CoherenceMatrix)
    assert isinstance(mixing_matrix, MixingMatrix.MixingMatrix)

def test_generate_signals_invalid_processing(setup_params):
    input_signals = np.random.randn(4, 16000)
    with pytest.raises(ValueError):
        generate_signals(input_signals, setup_params, decomposition='evd', processing='invalid')

def test_generate_signals_corcos(setup_params_corcos):
    input_signals = np.random.randn(4, 16000)
    output_signals, coherence_target, mixing_matrix = generate_signals(
        input_signals, setup_params_corcos, decomposition='evd', processing='standard'
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

def test_generate_signals_invalid_sc_type(setup_params):
    setup_params.sc_type = 'invalid'
    input_signals = np.random.randn(4, 16000)
    with pytest.raises(ValueError):
        generate_signals(input_signals, setup_params, decomposition='evd', processing='standard')

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

def test_generate_signals_eval(setup_params):
    input_signals = np.random.randn(4, 16000)
    output_signals, coherence_target, mixing_matrix = generate_signals(
        input_signals, setup_params, decomposition='evd', processing='balance+smooth'
    )
    mixing_matrix.evaluate_matrix(plot=True)
    estimate_coherence(output_signals, setup_params.nfft)
    assert output_signals.shape == input_signals.shape
    assert isinstance(coherence_target, CoherenceMatrix.CoherenceMatrix)
    assert isinstance(mixing_matrix, MixingMatrix.MixingMatrix)

def test_generate_signals_eval_corcos(setup_params_corcos):
    input_signals = np.random.randn(4, 16000)
    output_signals, coherence_target, mixing_matrix = generate_signals(
        input_signals, setup_params_corcos, decomposition='evd', processing='balance+smooth'
    )
    mixing_matrix.evaluate_matrix(plot=True)
    estimate_coherence(output_signals, setup_params_corcos.nfft)
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
