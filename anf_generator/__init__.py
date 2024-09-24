import numpy as np
import scipy.linalg
from . import MixingMatrix
from . import CoherenceMatrix

def generate_signals(input_signals, params, decomposition='evd', processing='balance+smooth'):
    """
    Generate M output signals with the desired spatial coherence.
    
    Input:
        input_signals: M mutually independent input signals [Channels x Samples]
        decomposition: Type of decomposition: 'evd' or 'chd'
        processing: Type of processing: 'standard', 'smooth', 'balance', 'balance+smooth'
        params: Parameters object
    
    Output:
        output_signals: Generated output signals [Channels x Samples]
        coherence_target: Target coherence matrix
        mixing_matrix: Mixing matrix object
    """

    # Generate target spatial coherence
    coherence_target = CoherenceMatrix.CoherenceMatrix(params)

    # Compute mixing matrix
    mixing_matrix = MixingMatrix.MixingMatrix(
        coherence_matrix=coherence_target,
        decomposition=decomposition,
        processing=processing
    )

    # Generate sensor signals with target spatial coherence
    output_signals = mix_signals(input_signals, mixing_matrix)

    return output_signals, coherence_target, mixing_matrix


def mix_signals(y, mixing_matrix: MixingMatrix.MixingMatrix):
    """
    Mix M mutually independent signals such that the mixed signals
    exhibit a specific spatial coherence.

    Parameters:
        n (numpy.array): M signals in the time domain [Channels x Samples]
        mixing_matrix (MixingMatrix): Mixing matrix object

    Returns:
        x (numpy.array): M generated signals [Channels x Samples]
    """

    # Initialization
    L = y.shape[1]  # Length input signal
    K = mixing_matrix.coherence_matrix.params.nfft  # FFT length

    # Compute STFT of the input signals
    _, _, Y = scipy.signal.stft(y, fs=1.0, window='hann', nperseg=K, noverlap=K//4, nfft=K,
                                return_onesided=True, boundary='zeros', padded=True, axis=-1,
                                scaling='spectrum')
    Y = np.transpose(Y, axes=[0, 2, 1]) # Transpose to [Channels x Samples x Frequencies]

    # Initialize STFT output
    X = np.zeros_like(Y)

    # Filter input signals
    for k in range(K//2, 0, -1):
        X[:, :, k] = np.conj(mixing_matrix.matrix[:, :, k]).T @ Y[:, :, k]

    # Inverse STFT
    X = np.transpose(X, axes=[0, 2, 1]) # Transpose to [Channels x Frequencies x Samples]
    _, x = scipy.signal.istft(X, fs=1.0, window='hann', nperseg=K, noverlap=K//4, nfft=K,
                              input_onesided=True, time_axis=-1, freq_axis=-2, scaling='spectrum')
    x = x[:, :L]  # Output signals

    return x


def estimate_coherence(x, nfft):
    """
    Compute the complex spatial coherence of x using the Welch's periodogram approach.

    Parameters:
        x (numpy.array): Multi-channel time signals [Channels x Samples]
        nfft (int): FFT length

    Returns:
        CC (numpy.array): Estimated complex coherence matrix [Channels x Channels x Frequencies]
    """
    if x.shape[1] == 1:
        raise ValueError(
            'Number of channels must be > 1 to compute the spatial coherence.')

    # Compute STFT with 75% overlap of the input
    _, _, X = scipy.signal.stft(x, fs=1.0, window='hann', nperseg=nfft, noverlap=nfft//4, 
                                nfft=nfft, return_onesided=True, boundary='zeros', 
                                padded=True, axis=-1, scaling='spectrum')

    M, K, L = X.shape  # Number of channels, frequencies, and frames

    # Compute PSD matrix
    psd_matrix = np.zeros((M, M, K), dtype=np.complex_)
    for l in range(L):
        # Compute instantaneous spectrum for the l-th frame
        Xf = X[:, :, l].T

        # Compute instantaneous PSD matrix
        XX = np.einsum('ij,ik->jki', Xf, np.conj(Xf))

        # Update
        psd_matrix += XX

    # Compute complex coherence matrix
    CC = np.zeros((M, M, K), dtype=np.complex_)
    for r in range(M):
        for c in range(M):
            CC[r, c, :] = psd_matrix[r, c, :] / \
                np.sqrt(psd_matrix[r, r, :] * psd_matrix[c, c, :])

    return CC
