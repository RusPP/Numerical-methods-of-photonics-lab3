import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import time


def main():
    step = np.array([1000, 500, 100])
    for item in step:
        start = time.time()
        # region Initial conditions
        _time = np.linspace(1, 1000, item)
        # selectable parameters
        tau = 500
        _w0 = 0.2
        _d = 0.05
        _g = 1e-3
        _c = 100
        phi = 2 * np.pi * _time * _d * np.sin(_time * _g)
        exp_part = (-(_time - tau) ** 2) / (2 * _c ** 2)
        amp = np.exp(exp_part)
        modulating_signal = amp * np.cos(_w0 * _time + phi)
        # endregion

        # region Without noise
        sp = plt.subplot(231)
        plt.plot(_time, phi)
        plt.title(r'Phase')
        plt.grid(True)

        # hilbert transform
        analytical_signal = hilbert(modulating_signal)

        # restored phase signal
        sp = plt.subplot(232)
        plt.plot(_time, np.unwrap(np.angle(analytical_signal)))
        plt.xlim(100, 900)
        plt.title(r'Restored phase')
        plt.grid(True)

        # original signal ( c(x) )
        sp = plt.subplot(233)
        plt.plot(_time, np.real(analytical_signal))
        plt.title(r'Original signal')
        plt.grid(True)

        # addition ( s(x) )
        sp = plt.subplot(234)
        plt.plot(_time, np.imag(analytical_signal))
        plt.title(r'Addition')
        plt.grid(True)

        # envelope ( |z(x)| )
        sp = plt.subplot(235)
        plt.plot(_time, np.real(analytical_signal), _time, np.abs(analytical_signal))
        plt.title(r'Original signal and envelope')
        plt.grid(True)

        # original and addition signals comparison
        sp = plt.subplot(236)
        plt.plot(_time, np.real(analytical_signal), _time, np.imag(analytical_signal))
        plt.title(r'Original signal and addition')
        plt.grid(True)
        # endregion

        part_1 = time.time() - start
        # show results
        plt.show()
        start = time.time()

        # region Add noise
        # add some noise
        noise_phase = phi + 0.5 * np.random.sample(len(phi))
        amp = np.exp(exp_part)
        modulating_signal = amp * np.cos(_w0 * _time + noise_phase)
        # Use hilbert transform
        noise_signal = hilbert(modulating_signal)

        sp = plt.subplot(221)
        plt.plot(_time, noise_phase)
        plt.title(r'Phase with noise')
        plt.grid(True)

        sp = plt.subplot(222)
        plt.plot(_time, np.unwrap(np.angle(noise_signal)))
        plt.xlim(100, 900)
        plt.title(r'Restored phase with noise')
        plt.grid(True)

        # noise and addition
        sp = plt.subplot(223)
        plt.plot(_time, np.real(noise_signal), _time, np.imag(noise_signal))
        plt.title(r'Restored and addition')
        plt.grid(True)

        # noise and envelop
        sp = plt.subplot(224)
        plt.plot(_time, np.real(noise_signal), _time, np.abs(noise_signal))
        plt.title(r'Restored and envelop')
        plt.grid(True)
        # endregion

        part_2 = time.time() - start
        alg_runtime = part_1 + part_2
        plt.show()
        print(f"With N = {item} algorithm runtime equals: {alg_runtime} sec")


if __name__ == '__main__':
    main()
