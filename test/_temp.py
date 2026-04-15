from typing import Dict, List, Optional, Tuple

import numpy as np


class DerivativeBasedVAD:
    """
    Version 3 - True Hybrid Derivative VAD (Recommended)
    Uses derivatives from both speech probabilities and RMS energy for better robustness.
    """

    def __init__(
        self,
        activation_th: float = 0.44,
        min_speech_frames: int = 1,
        base_alpha: float = 0.47,
        delta_weight: float = 0.95,
        onset_boost_base: float = 0.25,
        raw_peak_th: float = 0.78,
        energy_weight: float = 0.40,  # 0.0 = probability only, 0.5 = strong energy influence
        verbose: bool = True,
    ):
        self.activation_th = activation_th
        self.min_speech_frames = min_speech_frames
        self.base_alpha = base_alpha
        self.delta_weight = delta_weight
        self.onset_boost_base = onset_boost_base
        self.raw_peak_th = raw_peak_th
        self.energy_weight = np.clip(energy_weight, 0.0, 0.7)
        self.verbose = verbose

    def _compute_delta(self, features: np.ndarray) -> np.ndarray:
        """First-order regression delta with window=2."""
        if features.ndim == 1:
            features = features.reshape(1, -1)
        n = features.shape[1]
        delta = np.zeros_like(features, dtype=float)
        denom = 2 * (1 + 4)

        for t in range(n):
            num = 0.0
            for k in range(1, 3):
                tp = min(t + k, n - 1)
                tm = max(t - k, 0)
                num += k * (features[:, tp] - features[:, tm])
            delta[:, t] = num / denom
        return delta[0] if features.shape[0] == 1 else delta

    def process(
        self, speech_probs: np.ndarray, rms_energy: Optional[np.ndarray] = None
    ) -> Dict:
        probs = np.asarray(speech_probs, dtype=float)
        rms = np.asarray(rms_energy, dtype=float) if rms_energy is not None else None

        if self.verbose:
            print("=== DerivativeBasedVAD v3 Hybrid Debug ===")

        delta_probs = self._compute_delta(probs)
        delta_rms = (
            self._compute_delta(rms) if rms is not None else np.zeros_like(probs)
        )

        smoothed = self._hybrid_adaptive_smooth(probs, rms, delta_probs, delta_rms)

        decisions = np.zeros(len(probs), dtype=int)

        if self.verbose:
            print(
                "\nFrame | RawProb | Smoothed | ΔProb    | ΔRMS     | EffTh  | HybridScore | Dec | Note"
            )

        for t in range(len(probs)):
            eff_th = self.activation_th
            boost = 0.0

            if t > 0 and delta_probs[t] > 0.05:
                boost = self.onset_boost_base * min(3.0, delta_probs[t] / 0.10)
                eff_th = max(0.25, self.activation_th - boost)

            is_raw_peak = probs[t] >= self.raw_peak_th

            # Hybrid score
            prob_score = smoothed[t]
            energy_score = rms[t] if rms is not None else 1.0
            hybrid_score = (
                1 - self.energy_weight
            ) * prob_score + self.energy_weight * min(1.0, energy_score * 10.0)

            decision = 1 if (hybrid_score >= eff_th or is_raw_peak) else 0
            decisions[t] = decision

            note = ""
            if decision == 1:
                if is_raw_peak:
                    note = "RAW PEAK"
                elif boost > 0.15:
                    note = "ONSET BOOST"
                else:
                    note = "HYBRID"

            if self.verbose:
                rms_str = f"{rms[t]:.4f}" if rms is not None else "N/A"
                print(
                    f"{t:3d} | {probs[t]:.4f} | {smoothed[t]:.4f} | {delta_probs[t]:+8.4f} | "
                    f"{delta_rms[t]:+8.4f} | {eff_th:.3f} | {hybrid_score:.4f} | {decision}   | {note}"
                )

        # Post-processing with tighter hangover control
        final = decisions.copy()

        i = 0
        while i < len(final):
            if final[i] == 1:
                j = i
                while j < len(final) and final[j] == 1:
                    j += 1
                length = j - i

                if length < self.min_speech_frames:
                    final[i:j] = 0
                elif length <= 6:
                    final[j : j + 1] = 0  # minimal hangover for short bursts
                i = j
            else:
                i += 1

        # Extract segments
        segments: List[Tuple[int, int]] = []
        i = 0
        while i < len(final):
            if final[i] == 1:
                start = i
                while i < len(final) and final[i] == 1:
                    i += 1
                segments.append((start, i))
            else:
                i += 1

        if self.verbose:
            print("\n=== FINAL RESULT ===")
            print("Speech segments:", segments)
            print("Total speech frames:", final.sum())

        return {
            "speech_segments": segments,
            "final_decisions": final,
            "smoothed_probs": smoothed,
            "delta_probs": delta_probs,
            "original_probs": probs,
        }

    def _hybrid_adaptive_smooth(
        self,
        probs: np.ndarray,
        rms: Optional[np.ndarray],
        delta_probs: np.ndarray,
        delta_rms: np.ndarray,
    ) -> np.ndarray:
        """Hybrid smoothing using derivatives from both probability and energy."""
        smoothed = np.zeros_like(probs)
        smoothed[0] = probs[0]

        if rms is None:
            change_norm = np.abs(delta_probs) / (np.max(np.abs(delta_probs)) + 1e-8)
        else:
            combined = np.abs(delta_probs) + self.energy_weight * 4.0 * np.abs(
                delta_rms
            )
            change_norm = combined / (np.max(combined) + 1e-8)

        for t in range(1, len(probs)):
            alpha = self.base_alpha - self.delta_weight * change_norm[t - 1]
            alpha = np.clip(alpha, 0.04, 0.80)
            smoothed[t] = alpha * probs[t] + (1 - alpha) * smoothed[t - 1]

        return smoothed


# ====================== Usage Example ======================
if __name__ == "__main__":
    np.random.seed(42)

    speech_probs = np.concatenate(
        [
            np.full(10, 0.08),
            np.linspace(0.12, 0.92, 8),
            np.full(25, 0.94) + np.random.normal(0, 0.035, 25),
            np.linspace(0.90, 0.18, 9),
            np.full(15, 0.09) + np.random.normal(0, 0.02, 15),
            np.linspace(0.15, 0.85, 6),
            np.full(7, 0.07),
        ]
    )

    rms_energy = np.concatenate(
        [
            np.full(10, 0.012),
            np.linspace(0.015, 0.13, 8),
            np.full(25, 0.125) + np.random.normal(0, 0.01, 25),
            np.linspace(0.13, 0.022, 9),
            np.full(15, 0.014),
            np.linspace(0.018, 0.10, 6),
            np.full(7, 0.011),
        ]
    )

    vad = DerivativeBasedVAD(verbose=True, energy_weight=0.40)
    result = vad.process(speech_probs, rms_energy)
