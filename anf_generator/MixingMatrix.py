import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from . import CoherenceMatrix


class MixingMatrix:
    def __init__(
        self, coherence_matrix: CoherenceMatrix.CoherenceMatrix, decomposition: str, processing: str
    ):
        self.coherence_matrix = coherence_matrix
        self.decomposition = decomposition
        self.processing = processing

        self.matrix, self.matrix_std = self.compute_mixing_matrix()

    def compute_mixing_matrix(self):
        # Initialization
        M = self.get_num_channels()  # Number of channels
        K = self.get_nfft()  # Number of frequency bins
        C = np.zeros((M, M, K), dtype=np.complex_)  # STFT mixing matrix

        # Direct Current component definition for the mixing matrix (fully coherent)
        C[:, :, 0] = np.ones((M, M)) / np.sqrt(M)

        # Generate mixing matrix in the STFT domain for each frequency bin k
        DC = self.coherence_matrix.matrix  # Target coherence matrix
        for k in range(K // 2, 0, -1):
            # Rank of the target coherence matrix
            DC_rank = np.linalg.matrix_rank(DC[:, :, k])

            if self.decomposition == "chd" and DC_rank == M:
                C[:, :, k] = scipy.linalg.cholesky(DC[:, :, k], lower=False)
            elif self.decomposition == "evd" or (
                self.decomposition == "chd" and DC_rank < M
            ):
                D, V = np.linalg.eigh(DC[:, :, k])
                D = np.clip(D, 0, np.inf)
                # np.dot(np.diag(np.sqrt(D)),V.conj().T)
                C[:, :, k] = np.sqrt(D)[:, np.newaxis] * V.conj().T
            else:
                raise ValueError(
                    'Unknown decomposition method specified. Please select "CHD" or "EVD".'
                )

        # Save original CHD/EVD mixing matrix for evaluation
        C_std = np.copy(C)

        # Apply selected processing method
        for k in range(K // 2, 0, -1):
            if self.processing == "debug":  # Only for testing STFT and iSTFT
                C[:, :, k] = np.eye(M)
            elif self.processing == "standard":
                continue
            elif self.processing == "balance":
                C[:, :, k] = MixingMatrix.balance(C[:, :, k])
            elif self.processing == "smooth":
                C[:, :, k - 1] = MixingMatrix.smooth(C[:, :, k], C[:, :, k - 1])
            elif self.processing == "balance+smooth":
                if k == K // 2:
                    C[:, :, k] = MixingMatrix.balance(C[:, :, k], U_type="orthogonal")
                C[:, :, k - 1] = MixingMatrix.smooth(C[:, :, k], C[:, :, k - 1])
                C[:, :, k - 1] = MixingMatrix.balance_perserving_smoothness(
                    C[:, :, k - 1]
                )
            else:
                raise ValueError(
                    'Unknown processing method specified. Please select "standard", "balance", "smooth", or "balance+smooth".'
                )

        return C, C_std

    def evaluate_matrix(self, plot=True):
        M = self.get_num_channels()  # Number of channels
        K = self.get_nfft()

        C_std = self.matrix_std[:, :, 0 : K // 2]
        C = self.matrix[:, :, 0 : K // 2]

        # Compute frequency-wise balance
        bal_std = np.sum(np.abs(C_std), axis=(0, 1)) / (M * np.sqrt(M))
        bal = np.sum(np.abs(C), axis=(0, 1)) / (M * np.sqrt(M))

        # Compute mean balance in dB
        bal_std_mean = 20 * np.log10(np.mean(bal_std))
        bal_mean = 20 * np.log10(np.mean(bal))

        # Compute frequency-wise smoothness
        smooth_std = np.sum(np.abs(np.diff(C_std, axis=2)) ** 2, axis=(0, 1))
        smooth = np.sum(np.abs(np.diff(C, axis=2)) ** 2, axis=(0, 1))

        # Compute mean smoothness in dB
        smooth_std_mean = 10 * np.log10(np.mean(smooth_std))
        smooth_mean = 10 * np.log10(np.mean(smooth))

        # Compute coherence error and IRs
        xi_std, xi, c_std, c = self.coherence_error()

        eval_metrics = {
            "balance_std": bal_std_mean,
            "smoothness_std": smooth_std_mean,
            "balance": bal_mean,
            "smoothness": smooth_mean,
            "coh_error_std": xi_std,
            "coh_error": xi,
        }

        if plot:
            self.__plot_smoothness_balance(
                smooth, smooth_std, bal, bal_std, eval_metrics
            )
            self.__plot_filters(c_std, c)

        return eval_metrics

    def coherence_error(self, plot=True):
        C1_std = self.matrix_std
        C1 = self.matrix

        K1 = self.get_nfft()  # Current DFT length
        K2 = int(np.floor(np.pi * K1))  # Increased DFT length K2 > K1
        M = self.get_num_channels()  # Number of channels

        # Pre-allocate memory
        c1_std = np.zeros((M, M, K1), dtype=np.complex_)
        c1 = np.zeros((M, M, K1), dtype=np.complex_)
        C2_std = np.zeros((M, M, K2), dtype=np.complex_)
        C2 = np.zeros((M, M, K2), dtype=np.complex_)

        for p in range(M):
            for q in range(M):
                # Inverse DFT to get IR
                c1_std[p, q, :] = np.fft.irfft(C1_std[p, q, :], n=K1)
                c1[p, q, :] = np.fft.irfft(C1[p, q, :], n=K1)

                # Circular shift
                c1_std[p, q, :] = np.roll(c1_std[p, q, :], K1 // 2)
                c1[p, q, :] = np.roll(c1[p, q, :], K1 // 2)

                # DFT with increased frame length K2
                C2_std[p, q, :] = np.fft.fft(c1_std[p, q, :], n=K2)
                C2[p, q, :] = np.fft.fft(c1[p, q, :], n=K2)

        # Generate target spatial coherence with DFT length K2 > K1
        params2 = self.coherence_matrix.params.copy()
        params2.nfft = K2
        DC2 = CoherenceMatrix.CoherenceMatrix(params2)

        # Compute generated coherence with DFT-length K2 as C'*C and nMSE
        G_std = np.zeros((M, M, K2 // 2 + 1), dtype=np.complex_)
        G = np.zeros((M, M, K2 // 2 + 1), dtype=np.complex_)
        nMSEk_std = np.zeros(K2 // 2 + 1)
        nMSEk = np.zeros(K2 // 2 + 1)
        for k in range(K2 // 2 + 1):
            G_std[:, :, k] = C2_std[:, :, k].conj().T @ C2_std[:, :, k]
            G[:, :, k] = C2[:, :, k].conj().T @ C2[:, :, k]
            nMSEk_std[k] = (
                np.linalg.norm(G_std[:, :, k] - DC2.matrix[:, :, k], "fro") ** 2
            )
            nMSEk[k] = np.linalg.norm(G[:, :, k] - DC2.matrix[:, :, k], "fro") ** 2

        # Compute normalized mean squared error between target and generated coherence
        xi_std = 10 * np.log10(np.mean(nMSEk_std))
        xi = 10 * np.log10(np.mean(nMSEk))

        if plot:
            self.__plot_coherence(G, G_std, DC2.matrix, xi, xi_std)

        return xi_std, xi, c1_std, c1

    def get_nfft(self):
        return self.coherence_matrix.params.nfft

    def get_num_channels(self):
        return len(self.coherence_matrix.params.mic_positions)

    def __plot_smoothness_balance(self, smooth, smooth_std, bal, bal_std, eval_metrics):
        # Define labels for ledgend
        smoothness_label = (f'{self.decomposition} {self.processing}: \n'
                            f'ε = {eval_metrics["smoothness"]:.1f} dB, '
                            f'β = {eval_metrics["balance"]:.1f} dB')
        smoothness_std_label = (f'{self.decomposition}: \n'
                                f'ε = {eval_metrics["smoothness_std"]:.1f} dB, '
                                f'β = {eval_metrics["balance_std"]:.1f} dB')

        # Plot Spectral Smoothness and Mix Balance
        plt.figure()

        plt.subplot(2, 1, 1)
        plt.plot(smooth, linewidth=2, label=smoothness_label)
        plt.plot(smooth_std, linestyle="-.", linewidth=2, label=smoothness_std_label)
        plt.legend()
        plt.grid(True)
        plt.ylabel("Smoothness")
        plt.ylim(0, np.amax(np.concatenate((smooth, smooth_std))))

        plt.subplot(2, 1, 2)
        plt.plot(bal, linewidth=2)
        plt.plot(bal_std, linestyle="-.", linewidth=2)
        plt.grid(True)
        plt.xlabel("Frequency")
        plt.ylabel("Balance")
        plt.ylim(0, 1)

    def __plot_coherence(self, G, G_std, DC2, xi, xi_std):
        K2 = DC2.shape[2]  # Increased DFT length
        sample_frequency = (
            self.coherence_matrix.params.sample_frequency
        )  # Sample frequency
        # Frequency vector in Hz
        Freqs = np.linspace(0, sample_frequency / 2, K2)

        # Procesd the decomposition and processing strings
        decomposition = self.decomposition
        processing = self.processing

        fig = plt.figure()
        plt.subplots_adjust(hspace=0.6)

        if (
            self.coherence_matrix.params.sc_type == "spherical"
            or self.coherence_matrix.params.sc_type == "cylindrical"
        ):
            # Case diffuse (real-valued coherence)
            plt.subplot(2, 1, 1)
            plt.plot(Freqs / 1000, np.real(G_std[1, 2, :]), "-k", linewidth=2)
            plt.plot(Freqs / 1000, np.real(DC2[1, 2, :]), "-.b", linewidth=2)
            plt.title(decomposition)
            plt.grid(True)
            plt.xlim([0, sample_frequency / 2000])
            plt.ylim([-1, 1])
            plt.ylabel("Spatial coherence")
            plt.legend([rf"$\xi$ = {xi_std:.1f} dB", "Target"])

            plt.subplot(2, 1, 2)
            plt.plot(Freqs / 1000, np.real(G[1, 2, :]), "-k", linewidth=2)
            plt.plot(Freqs / 1000, np.real(DC2[1, 2, :]), "-.b", linewidth=2)
            plt.title(f"{decomposition} {processing}")
            plt.grid(True)
            plt.xlim([0, sample_frequency / 2000])
            plt.ylim([-1, 1])
            plt.ylabel("Spatial coherence")
            plt.legend([rf"$\xi$ = {xi:.1f} dB", "Target"])

            fig.suptitle("Coherence error - Sensors 1-2")
            fig.supxlabel("Frequency [kHz]")
        else:
            # Case 'corcos' or generally complex-valued coherence
            plt.subplot(2, 2, 1)
            plt.plot(Freqs / 1000, np.real(G_std[1, 2, :]), "-k", linewidth=2)
            plt.plot(Freqs / 1000, np.real(DC2[1, 2, :]), "-.b", linewidth=2)
            plt.title(decomposition)
            plt.grid(True)
            plt.xlim([0, sample_frequency / 2000])
            plt.ylim([-1, 1])
            plt.ylabel("Real")
            plt.legend([rf"$\xi$ = {xi_std:.1f} dB", "Target"])

            plt.subplot(2, 2, 2)
            plt.plot(Freqs / 1000, np.imag(G_std[1, 2, :]), "-k", linewidth=2)
            plt.plot(Freqs / 1000, np.imag(DC2[1, 2, :]), "-.b", linewidth=2)
            plt.xlim([0, sample_frequency / 2000])
            plt.ylim([-1, 1])
            plt.ylabel("Imag")
            plt.grid(True)

            plt.subplot(2, 2, 3)
            plt.plot(Freqs / 1000, np.real(G[1, 2, :]), "-k", linewidth=2)
            plt.plot(Freqs / 1000, np.real(DC2[1, 2, :]), "-.b", linewidth=2)
            plt.title(f"{decomposition} {processing}")
            plt.grid(True)
            plt.xlim([0, sample_frequency / 2000])
            plt.ylim([-1, 1])
            plt.legend([rf"$\xi$ = {xi:.1f} dB", "Target"])

            plt.subplot(2, 2, 4)
            plt.plot(Freqs / 1000, np.imag(G[1, 2, :]), "-k", linewidth=2)
            plt.plot(Freqs / 1000, np.imag(DC2[1, 2, :]), "-.b", linewidth=2)
            plt.xlim([0, sample_frequency / 2000])
            plt.ylim([-1, 1])
            plt.grid(True)

            fig.suptitle("Coherence error - Sensors 1-2")
            fig.supylabel("Spatial Coherence")
            fig.supxlabel("Frequency [kHz]")

        # plt.show()

    def __plot_filters(self, c_std, c):
        # Choose which IRs of the mixing matrix to plot
        row = 1
        column = 1

        # Normalize IRs
        c_std_normalized = np.abs(c_std[row, column, :]) / np.max(
            np.abs(c_std[row, column, :])
        )
        c_normalized = np.abs(c[row, column, :]) / np.max(np.abs(c[row, column, :]))

        # Plot IRs of the mixing matrix
        fig = plt.figure()

        plt.subplot(1, 2, 1)
        plt.plot(20 * np.log10(c_std_normalized), linewidth=1)
        plt.title(f"{self.decomposition} - IR({row+1},{column+1})")
        plt.grid(True)
        plt.xlabel("Time")

        plt.subplot(1, 2, 2)
        plt.plot(20 * np.log10(c_normalized), linewidth=1)
        plt.title(f"{self.decomposition} {self.processing} - IR({row+1},{column+1})")
        plt.grid(True)
        plt.xlabel("Time")

        fig.supylabel("Magnitude [dB]")

    @staticmethod
    def balance(C, U_type="unitary"):
        """
        Compute optimal balance B = U * C with max l1(U * C) for U
        unitary or orthogonal.

        Parameters:
            C (numpy.array): Mixing matrix [Channels x Channels]
            type: 'unitary' (default) for U unitary, and 'orthogonal' for U orthogonal

        Returns:
            B (numpy.array): balance mixing matrix [Channels x Channels]
        """
        num_channels = C.shape[0]  # Number of channels

        # Init with a matrix of scaled ones
        H = np.ones((num_channels, num_channels)) / np.sqrt(num_channels)

        # Perform optimization
        if U_type == "unitary":
            U = MixingMatrix.absoluteUnitaryProcrustes(C, H)
        elif U_type == "orthogonal":
            U = MixingMatrix.absoluteOrthogonalProcrustes(C, H)

        # Compute balanced matrix
        B = U @ C

        return B

    @staticmethod
    def absoluteUnitaryProcrustes(A, B):
        """
        Solves || abs(U * A) - abs(B) ||_F with unitary U.
        """
        MaximumTrails = 10  # Number of trials
        N = A.shape[0]  # Number of channels
        newPhases = np.exp(1j * np.angle(B, deg=False))  # Phase matrix of B

        bestMatrix = None
        bestL1 = -np.inf

        for _ in range(MaximumTrails):
            # Perform optimization
            U = MixingMatrix.signVariableExchange(A, newPhases, False)

            # Evaluate l1-norm of obtained unitary matrix
            l1 = MixingMatrix.l1norm(U @ A)

            # Check if this is the best matrix so far
            if l1 > bestL1:
                bestL1 = l1
                bestMatrix = U

            # Generate new (random) normalized phase matrix
            newPhases = np.exp(1j * np.random.rand(N, N) * 2 * np.pi)

        return bestMatrix

    @staticmethod
    def absoluteOrthogonalProcrustes(A, B):
        """
        Solves || abs(U * A) - abs(B) ||_F with ortogonal U for
        A, B and U being real-valued.
        """
        MaximumTrails = 10  # Number of trials
        N = A.shape[0]  # Number of channels
        newSigns = np.sign(np.real(B) + np.finfo(float).eps)  # Sign matrix of B

        bestMatrix = None
        bestL1 = -np.inf

        for _ in range(MaximumTrails):
            # Perform optimization
            U = MixingMatrix.signVariableExchange(A, newSigns, False)

            # Evaluate l1-norm of obtained unitary matrix
            l1 = MixingMatrix.l1norm(U @ A)

            # Check if this is the best matrix so far
            if l1 > bestL1:
                bestL1 = l1
                bestMatrix = U

            # Generate new (random) normalized phase matrix
            newSigns = np.sign(np.random.rand(N, N) * 2 - 1)

        return bestMatrix

    @staticmethod
    def signVariableExchange(A, B, verbose=False):
        """
        Perform the variable exchange optimization.

        Parameters:
            A (numpy.array): First matrix
            B (numpy.array): Second matrix (Phase matrix)
            verbose (bool): If True, print all messages. If False, print only warning messages.

        Returns:
            U (numpy.array): Unitary matrix that minimizes (U*A - B) in the Frobenius norm
        """
        maxIter = 1000  # Max number of iterations
        bestFit = np.inf  # Best fit in the Frobenius-norm sense
        Phases = np.exp(1j * np.angle(B))  # Phase matrix of B

        newFit = np.zeros(maxIter)
        l1Fit = np.zeros(maxIter)
        # Us = []

        for counter in range(maxIter):
            # New phase matrix
            phaseB = Phases

            # Procrustes solution
            U = MixingMatrix.procrustes(A, phaseB)

            # Compute Frobenius norm of (U*A - phaseB)
            newFit[counter] = np.linalg.norm((U @ A) - phaseB, "fro")

            delta = newFit[counter] - bestFit

            # Update bestFit
            bestFit = newFit[counter]

            # Store l1norm
            l1Fit[counter] = MixingMatrix.l1norm(U @ A)

            # Store the unitary matrix
            # Us.append(U)

            # Check convergence
            if delta > np.sqrt(np.finfo(float).eps):
                if verbose:
                    print("Variable exchange optimization did not improve.")
            elif delta > -1e-5:
                if verbose:
                    print(
                        f"Variable exchange optimization converged within {counter} iterations."
                    )
                break

            # Update phase matrix
            Phases = np.exp(1j * np.angle(U @ A))

            if counter == maxIter - 1:
                print(
                    f"WARNING: Variable exchange optimization did not converge within "
                    f"{maxIter} iterations."
                )

        return U

    @staticmethod
    def procrustes(A, B):
        """
        Compute nearest orthogonal matrix U (in the Frobenius norm) via SVD such
        that it minimizes U*A - B.

        Parameters:
            A (numpy.array): First matrix
            B (numpy.array): Second matrix

        Returns:
            U (numpy.array): Unitary matrix that minimizes (U*A - B) in the Frobenius norm
        """
        # Compute SVD of B @ A.T
        W, _, V = np.linalg.svd(B @ A.conj().T)

        # Compute orthogonal polar factor of (B @ A.T)
        U = W @ V

        return U

    @staticmethod
    def smooth(C, C_plus):
        """
        Compute nearest UNITARY matrix U (in the Frobenius norm) via Procrustes
        solution such that (U*C_plus - C) is minimized. The output is the updated
        matrix C_n = U*C_plus.

        Parameters:
            A (numpy.array): Mixing matrix [Channels x Channels]
            B (numpy.array): Another mixing matrix [Channels x Channels]

        Returns:
            C_smooth (numpy.array): Smooth (w.r.t. neighbor C) mixing matrix [Channels x Channels]
        """
        U = MixingMatrix.procrustes(C_plus, C)
        C_smooth = U @ C_plus

        return C_smooth

    @staticmethod
    def balance_perserving_smoothness(C):
        """
        Compute balance Cbs = U * C with high l1(U * C) for U UNITARY while
        preserving smoothness (=single iteration of balance.m with closest
        phases of C).

        Parameters:
            C (numpy.array): Mixing matrix [Channels x Channels]

        Returns:
            Cbs (numpy.array): Smoothness-preserving balance mixing matrix [Channels x Channels]
        """
        # Closest phases (phases of C)
        B = np.exp(1j * np.angle(C, deg=False))

        # Procrustes solution
        U = MixingMatrix.procrustes(C, B)

        # Smoothing-preserving balanced matrix
        Cbs = U @ C

        return Cbs

    @staticmethod
    def l1norm(X):
        """
        Normalized l1 norm (element-wise)
        For ||X||_F = 1, the norm is in [0,1].
        """
        N = X.shape[0]

        return np.sum(np.abs(X)) / (N * np.sqrt(N))
