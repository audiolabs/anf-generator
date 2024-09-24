"""
Example to generate multichannel babble speech with a desired spatial coherence.
"""

import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import anf_generator as anf


def plot_spatial_coherence(coherence_gen, coherence_target, params):
    num_channels = len(params.mic_positions)
    sample_frequency = params.sample_frequency
    nfft = params.nfft

    # Plot example of generated vs target spatial coherence
    freqs = np.linspace(0, sample_frequency / 2, nfft // 2 + 1) / 1000  # Freq. in kHz
    if params.sc_type in ["spherical", "cylindrical"]:
        fig = plt.figure()
        plt.subplots_adjust(hspace=0.5)
        fig.suptitle("Spatial Coherence")

        for m in range(num_channels - 1):
            plt.subplot(num_channels - 1, 1, m + 1)
            plt.plot(freqs, np.real(coherence_gen[0, m + 1, :]), "-k", linewidth=2)
            plt.plot(freqs, np.real(coherence_target[0, m + 1, :]), "-.b", linewidth=2)
            plt.title(f"Sensors 1-{m+2}")
            plt.axis([0, sample_frequency / 2000, -1, 1])
            plt.grid(True)
            plt.ylabel("Spatial coherence")

        fig.supxlabel("Frequency [kHz]")
    else:
        fig = plt.figure(constrained_layout=True)
        fig.suptitle("Spatial Coherence")

        subfigs = fig.subfigures(nrows=num_channels - 1, ncols=1)

        for row, subfig in enumerate(subfigs):
            subfig.suptitle(f"Sensors 1-{row+2}")

            axs = subfig.subplots(nrows=1, ncols=2)
            axs[0].plot(freqs, np.real(coherence_gen[0, row + 1, :]), "-k", linewidth=2)
            axs[0].plot(
                freqs, np.real(coherence_target[0, row + 1, :]), "-.b", linewidth=2
            )
            axs[0].grid(True)
            axs[0].set_ylim([-1, 1])
            axs[0].set_title("Real")

            axs[1].plot(freqs, np.imag(coherence_gen[0, row + 1, :]), "-k", linewidth=2)
            axs[1].plot(
                freqs, np.imag(coherence_target[0, row + 1, :]), "-.b", linewidth=2
            )
            axs[1].grid(True)
            axs[1].set_ylim([-1, 1])
            axs[1].set_title("Imaginary")

            if row == num_channels - 2:
                axs[0].set_xlabel("Frequency [kHz]")
                axs[1].set_xlabel("Frequency [kHz]")

    return


def main():
    # Set decomposition and processing methods
    decomposition = "evd"  # Type of decomposition: 'evd' or 'chd'
    processing = (
        "balance+smooth"  # Processing: 'standard', 'smooth', 'balance', 'balance+smooth'
    )
    duration = 10  # Input duration in seconds
    num_channels = 4  # Number of microphones
    evaluate = True # Set to True to evaluate results

    # Set parameters to define the target coherence
    params = anf.CoherenceMatrix.Parameters(
        mic_positions=np.array([[0.04 * i, 0, 0] for i in range(num_channels)]),
        # Type of Spatial Coherence: 'corcos', 'spherical', or 'cylindrical'
        sc_type="spherical",
        sample_frequency=8000,
        nfft=1024,
    )

    # Summarize main parameters
    print(f"Number of channels: {len(params.mic_positions)}")
    print(f"Spatial coherence: {params.sc_type}")
    print(f"Decomposition: {decomposition}")
    print(f"Processing: {processing}\n")

    # Read the WAV file
    sample_frequency_wav, input_signal = wavfile.read('babble_8kHz.wav')
    if sample_frequency_wav != params.sample_frequency:
        raise ValueError("The sample frequency of the input signal does not "
                         "match the desired sampling frequency.")

    # Calculate the number of samples
    num_samples = duration * sample_frequency_wav

    # Ensure the input signal is long enough
    if len(input_signal) < num_channels * num_samples:
        raise ValueError(f"Input signal is not long enough to extract {num_channels} "
                         f"non-overlapping segments of {duration} seconds each.")

    # Slice the input signal into non-overlapping segments
    trimmed_signal = input_signal[:num_channels * num_samples]
    input_signals = trimmed_signal.reshape(num_channels, num_samples)

    # Generate output signals with the desired spatial coherence
    output_signals, coherence_target, mixing_matrix = anf.generate_signals(
        input_signals, params, decomposition, processing)

    # Scale and save the generated signals to a WAV file
    scaled_signals = (output_signals / np.max(np.abs(output_signals)) * 32767).astype(np.int16)
    wavfile.write('mc_babble_8kHz.wav', params.sample_frequency, scaled_signals.T)

    if evaluate:
        # Evaluate mixing matrix
        eval_results = mixing_matrix.evaluate_matrix(plot=True)

        # Estimate coherence from the generated sensor signals
        coherence_gen = anf.estimate_coherence(output_signals, params.nfft)

        # Plot spatial coherence matrices
        plot_spatial_coherence(coherence_gen, coherence_target.matrix, params)

        # Compute error between target and generated coherence
        xi = np.sum(np.abs(coherence_gen - coherence_target.matrix) ** 2, axis=(0, 1))
        xi_avg = 10 * np.log10(np.average(xi))

        # Summarize evaluation results
        print("Improvements:")
        print(
            f'Spectral Variation: '
            f'{round(eval_results["smoothness"] - eval_results["smoothness_std"],2)} dB'
        )
        print(
            f'Mix Balance: '
            f'{round(eval_results["balance"] - eval_results["balance_std"],2)} dB'
        )
        print(
            f'Coherence Error: '
            f'{round(eval_results["coh_error"] - eval_results["coh_error_std"],2)} dB'
        )
        print(f"Average Coherence Error: "
              f"{round(xi_avg, 2)} dB\n")
        
        plt.show()


if __name__ == "__main__":
    main()
