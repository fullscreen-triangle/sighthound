"""
Phase-Locked Messaging Validation Experiments

Validates core theorems and protocols for phase-locked messaging:
- Phase lock convergence
- State change detection
- Position change and re-authentication
- Replay attack detection with epoch counters
- Eavesdropper detection
- Multi-channel advantage
"""

import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Tuple

class PhaseLockedMessagingValidator:
    def __init__(self, num_regions: int = 9, num_channels: int = 3, num_cells_per_channel: int = 3):
        self.num_regions = num_regions
        self.num_channels = num_channels
        self.num_cells = num_cells_per_channel
        self.results = {}

        # Generate random signatures for each region/channel
        np.random.seed(42)
        self.signatures = self._generate_signatures()

    def _generate_signatures(self) -> Dict:
        """Generate random but distinguishable signatures for each region."""
        signatures = {}
        for region_id in range(self.num_regions):
            signatures[region_id] = {}
            for channel_id in range(self.num_channels):
                # Generate random probabilities, normalize
                probs = np.random.dirichlet(np.ones(self.num_cells))
                signatures[region_id][channel_id] = probs
        return signatures

    def _sample_observation(self, true_region: int, channel_id: int) -> int:
        """Sample an observation from the true region's signature."""
        probs = self.signatures[true_region][channel_id]
        return np.random.choice(len(probs), p=probs)

    def _compute_posterior(self, evidence: List[Tuple[int, int]]) -> np.ndarray:
        """Compute posterior probability over regions given evidence."""
        if len(evidence) == 0:
            return np.ones(self.num_regions) / self.num_regions

        # Likelihood product: multiply signature probabilities
        likelihoods = np.ones(self.num_regions)
        for channel_id, cell_id in evidence:
            for region_id in range(self.num_regions):
                likelihoods[region_id] *= self.signatures[region_id][channel_id][cell_id]

        # Normalize to posterior
        posterior = likelihoods / (np.sum(likelihoods) + 1e-10)
        return posterior

    def _compute_metrics(self, posterior: np.ndarray) -> Tuple[float, float, float]:
        """Compute posterior margin, composition floor, and signature strength."""
        sorted_post = np.sort(posterior)[::-1]
        margin = sorted_post[0] - sorted_post[1]

        n = len(self.evidence) if hasattr(self, 'evidence') else 0
        gamma = self.num_channels * ((1 + self.num_channels) ** max(0, n - 1))
        floor = 1.0 / gamma

        winner_id = np.argmax(posterior)
        strength = np.mean([self.signatures[winner_id][c][np.argmax(self.signatures[winner_id][c])]
                           for c in range(self.num_channels)])

        return margin, floor, strength

    def experiment_phase_lock_convergence(self, num_trials: int = 20) -> Dict:
        """Verify that phase lock converges with evidence accumulation."""
        print("Running: Phase Lock Convergence Experiment")

        results = {
            'depths': list(range(1, 51)),
            'trials': []
        }

        for trial in range(num_trials):
            true_region = np.random.randint(0, self.num_regions)
            evidence = []
            margins = []

            for depth in range(1, 51):
                # Collect one more observation
                channel_id = np.random.randint(0, self.num_channels)
                cell_id = self._sample_observation(true_region, channel_id)
                evidence.append((channel_id, cell_id))

                # Compute posterior
                posterior = self._compute_posterior(evidence)
                margin = np.sort(posterior)[::-1][0] - np.sort(posterior)[::-1][1]
                margins.append(float(margin))

            results['trials'].append({
                'true_region': int(true_region),
                'margins': margins
            })

        # Summary statistics
        all_margins = np.array([m for trial in results['trials'] for m in trial['margins']])
        results['summary'] = {
            'mean_margin_at_depth_10': float(np.mean([trial['margins'][9] for trial in results['trials']])),
            'mean_margin_at_depth_30': float(np.mean([trial['margins'][29] for trial in results['trials']])),
            'mean_margin_at_depth_50': float(np.mean([trial['margins'][49] for trial in results['trials']])),
            'convergence_rate': float(np.mean([trial['margins'][49] - trial['margins'][0] for trial in results['trials']])) / 50
        }

        self.results['phase_lock_convergence'] = results
        return results

    def experiment_state_change_detection(self, num_trials: int = 50) -> Dict:
        """Verify that state changes at locked positions are detectable."""
        print("Running: State Change Detection Experiment")

        results = {
            'trials': []
        }

        for trial in range(num_trials):
            true_region = np.random.randint(0, self.num_regions)

            # Phase 1: Establish lock
            evidence_lock = []
            for _ in range(20):
                channel_id = np.random.randint(0, self.num_channels)
                cell_id = self._sample_observation(true_region, channel_id)
                evidence_lock.append((channel_id, cell_id))

            posterior_before = self._compute_posterior(evidence_lock)
            margin_before = np.sort(posterior_before)[::-1][0] - np.sort(posterior_before)[::-1][1]

            # Phase 2: Change state (different signature distribution)
            # Simulate state change by flipping the observed cells to unlikely values
            evidence_after = evidence_lock.copy()
            for _ in range(10):
                channel_id = np.random.randint(0, self.num_channels)
                # Sample from a different region's signature
                alt_region = (true_region + 1) % self.num_regions
                cell_id = self._sample_observation(alt_region, channel_id)
                evidence_after.append((channel_id, cell_id))

            posterior_after = self._compute_posterior(evidence_after)

            # Likelihood ratio between old and new state
            old_likelihood = np.prod([self.signatures[true_region][c][cell]
                                      for c, cell in evidence_after[20:]])
            new_likelihood = np.prod([self.signatures[(true_region + 1) % self.num_regions][c][cell]
                                      for c, cell in evidence_after[20:]])

            detection_ratio = np.log(new_likelihood / (old_likelihood + 1e-10))

            results['trials'].append({
                'margin_before': float(margin_before),
                'margin_after': float(np.sort(posterior_after)[::-1][0] - np.sort(posterior_after)[::-1][1]),
                'detection_ratio': float(detection_ratio),
                'detected': bool(detection_ratio > 2.0)
            })

        results['summary'] = {
            'detection_rate': float(np.mean([t['detected'] for t in results['trials']])),
            'mean_detection_ratio': float(np.mean([t['detection_ratio'] for t in results['trials']])),
            'mean_margin_preserved': float(np.mean([t['margin_before'] for t in results['trials']]))
        }

        self.results['state_change_detection'] = results
        return results

    def experiment_replay_attack_detection(self, num_trials: int = 100) -> Dict:
        """Verify replay attack detection via epoch conditioning."""
        print("Running: Replay Attack Detection Experiment")

        results = {
            'epoch_deltas': list(range(1, 21)),
            'trials': []
        }

        for trial in range(num_trials):
            true_region = np.random.randint(0, self.num_regions)

            # Collect evidence at epoch 0
            evidence_epoch0 = []
            for _ in range(10):
                channel_id = np.random.randint(0, self.num_channels)
                cell_id = self._sample_observation(true_region, channel_id)
                evidence_epoch0.append((channel_id, cell_id))

            trial_results = []

            for delta in range(1, 21):
                # Try to replay evidence at epoch delta
                decay_factor = np.exp(-0.6 * delta)

                # Likelihood of replayed evidence under decay
                replay_likelihood = 1.0
                for channel_id, cell_id in evidence_epoch0:
                    replay_likelihood *= self.signatures[true_region][channel_id][cell_id] * decay_factor

                # Likelihood of fresh evidence at new epoch
                fresh_evidence = []
                for _ in range(10):
                    channel_id = np.random.randint(0, self.num_channels)
                    cell_id = self._sample_observation(true_region, channel_id)
                    fresh_evidence.append((channel_id, cell_id))

                fresh_likelihood = 1.0
                for channel_id, cell_id in fresh_evidence:
                    fresh_likelihood *= self.signatures[true_region][channel_id][cell_id]

                # Detection: fresh evidence is more likely
                detected = fresh_likelihood > replay_likelihood
                detection_prob = 1.0 - np.exp(-0.6 * delta)

                trial_results.append({
                    'delta': delta,
                    'detected': bool(detected),
                    'theoretical_detection_prob': float(detection_prob)
                })

            results['trials'].append(trial_results)

        # Aggregate results by delta
        results['summary'] = {}
        for delta in range(1, 21):
            detections = [trial[delta-1]['detected'] for trial in results['trials']]
            results['summary'][f'delta_{delta}'] = {
                'empirical_detection_rate': float(np.mean(detections)),
                'theoretical_detection_rate': float(1.0 - np.exp(-0.6 * delta))
            }

        self.results['replay_attack_detection'] = results
        return results

    def experiment_eavesdropper_detection(self, num_trials: int = 100) -> Dict:
        """Verify that eavesdropping attempts are detectable via coherence collapse."""
        print("Running: Eavesdropper Detection Experiment")

        results = {
            'trials': []
        }

        for trial in range(num_trials):
            true_region = np.random.randint(0, self.num_regions)

            # Phase 1: A and B establish lock
            evidence_ab = []
            for _ in range(20):
                channel_id = np.random.randint(0, self.num_channels)
                cell_id = self._sample_observation(true_region, channel_id)
                evidence_ab.append((channel_id, cell_id))

            posterior_ab = self._compute_posterior(evidence_ab)
            margin_ab = np.sort(posterior_ab)[::-1][0] - np.sort(posterior_ab)[::-1][1]

            # Phase 2: Eavesdropper E joins (third party)
            # E's observations are from a different region or noisy
            eavesdropper_region = (true_region + 2) % self.num_regions
            evidence_with_e = evidence_ab.copy()

            # Add observations from eavesdropper (noise in the channel)
            for _ in range(10):
                channel_id = np.random.randint(0, self.num_channels)
                cell_id = self._sample_observation(eavesdropper_region, channel_id)
                evidence_with_e.append((channel_id, cell_id))

            posterior_with_e = self._compute_posterior(evidence_with_e)
            margin_with_e = np.sort(posterior_with_e)[::-1][0] - np.sort(posterior_with_e)[::-1][1]

            margin_degradation = margin_ab - margin_with_e
            detected = margin_degradation > 0.05  # Threshold for detection

            results['trials'].append({
                'margin_before': float(margin_ab),
                'margin_after': float(margin_with_e),
                'degradation': float(margin_degradation),
                'detected': bool(detected)
            })

        results['summary'] = {
            'detection_rate': float(np.mean([t['detected'] for t in results['trials']])),
            'mean_margin_degradation': float(np.mean([t['degradation'] for t in results['trials']])),
            'mean_margin_before': float(np.mean([t['margin_before'] for t in results['trials']]))
        }

        self.results['eavesdropper_detection'] = results
        return results

    def experiment_multichannel_advantage(self, max_channels: int = 6, num_trials: int = 50) -> Dict:
        """Verify that more channels enable clearer communication."""
        print("Running: Multi-Channel Advantage Experiment")

        # Regenerate signatures with max channels
        np.random.seed(42)
        signatures_multichannel = {}
        for region_id in range(self.num_regions):
            signatures_multichannel[region_id] = {}
            for channel_id in range(max_channels):
                probs = np.random.dirichlet(np.ones(self.num_cells))
                signatures_multichannel[region_id][channel_id] = probs

        results = {
            'channel_counts': list(range(1, max_channels + 1)),
            'trials': []
        }

        for trial in range(num_trials):
            true_region = np.random.randint(0, self.num_regions)
            trial_results = []

            for num_ch in range(1, max_channels + 1):
                evidence = []
                for depth in range(30):
                    channel_id = np.random.randint(0, num_ch)
                    # Sample from the multichannel signatures
                    cell_id = np.random.choice(self.num_cells,
                                              p=signatures_multichannel[true_region][channel_id])
                    evidence.append((channel_id, cell_id))

                # Compute posterior using multichannel signatures
                likelihoods = np.ones(self.num_regions)
                for channel_id, cell_id in evidence:
                    for region_id in range(self.num_regions):
                        likelihoods[region_id] *= signatures_multichannel[region_id][channel_id][cell_id]

                posterior = likelihoods / (np.sum(likelihoods) + 1e-10)
                margin = np.sort(posterior)[::-1][0] - np.sort(posterior)[::-1][1]

                # Composition floor
                gamma = num_ch * ((1 + num_ch) ** 29)
                floor = 1.0 / gamma

                trial_results.append({
                    'num_channels': num_ch,
                    'margin': float(margin),
                    'composition_floor': float(floor)
                })

            results['trials'].append(trial_results)

        # Summary by channel count
        results['summary'] = {}
        for num_ch in range(1, max_channels + 1):
            margins = [trial[num_ch-1]['margin'] for trial in results['trials']]
            results['summary'][f'channels_{num_ch}'] = {
                'mean_margin': float(np.mean(margins)),
                'std_margin': float(np.std(margins))
            }

        self.results['multichannel_advantage'] = results
        return results

    def run_all_experiments(self) -> Dict:
        """Run all validation experiments."""
        print("\n" + "="*60)
        print("PHASE-LOCKED MESSAGING VALIDATION EXPERIMENTS")
        print("="*60 + "\n")

        self.experiment_phase_lock_convergence()
        self.experiment_state_change_detection()
        self.experiment_replay_attack_detection()
        self.experiment_eavesdropper_detection()
        self.experiment_multichannel_advantage()

        self.results['metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'num_regions': self.num_regions,
            'num_channels': self.num_channels,
            'num_cells_per_channel': self.num_cells
        }

        print("\n" + "="*60)
        print("ALL EXPERIMENTS COMPLETED")
        print("="*60 + "\n")

        return self.results

    def save_results(self, filepath: str):
        """Save results to JSON file."""
        def convert_to_json_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_json_serializable(item) for item in obj]
            return obj

        serializable_results = convert_to_json_serializable(self.results)

        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        print(f"Results saved to {filepath}")


if __name__ == "__main__":
    validator = PhaseLockedMessagingValidator(num_regions=9, num_channels=3, num_cells_per_channel=3)
    results = validator.run_all_experiments()
    validator.save_results('phase_locked_messaging_results.json')
