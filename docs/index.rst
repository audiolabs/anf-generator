ANF Generator
===============================

.. toctree::
    :maxdepth: 2
    :hidden:

    references

.. image:: https://readthedocs.org/projects/anf-generator/badge/?version=latest
    :target: https://anf-generator.readthedocs.io/en/latest/?badge=latest

Python-based multisensor noise signal generator.

Install
-------

.. code-block:: bash

    pip install anf-generator
    
Example
-------

.. code-block:: python

    import numpy as np
    import anf_generator as anf

    # Define signals
    duration = 10  # Duration in seconds
    num_channels = 4  # Number of microphones

    # Define the target coherence
    params = anf.CoherenceMatrix.Parameters(
        mic_positions=np.array([[0.04 * i, 0, 0] for i in range(num_channels)]),
        sc_type="spherical",
        sample_frequency=16000,
        nfft=1024,
    )

    # Generate "num_channels" mutually independent input signals of length "duration"
    input_signals = np.random.randn(num_channels, duration * params.sample_frequency)

    # Generate output signals with the desired spatial coherence
    output_signals, coherence_target, mixing_matrix = anf.generate_signals(
        input_signals, params, decomposition='evd', processing='balance+smooth')
