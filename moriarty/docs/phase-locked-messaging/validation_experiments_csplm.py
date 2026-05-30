"""
Complementary-Strand Phase-Locked Messaging (CSPLM) Validation Experiments

Validates CSPLM-specific enhancements:
- Complement reconstruction accuracy
- Complement consistency verification
- Forging amplification (must forge both τ and τ̄)
- CSPLM vs PLM security comparison
- Complement integrity with noise detection
"""

import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Tuple

class CSPLMValidator:
    def __init__(self, num_regions: int = 9, num_channels: int = 3, num_cells_per_channel: int = 3):
        self.num_regions = num_regions
        self.num_channels = num_channels
        self.num_cells = num_cells_per_channel
        self.results = {}

        np.random.seed(42)
        self.signatures = self._generate_signatures()

    def _generate_signatures(self) -> Dict:
        """Generate random but distinguishable signatures for each region."""
        signatures = {}
        for region_id in range(self.num_regions):
            signatures[region_id] = {}
            for channel_id in range(self.num_channels):
                probs = np.random.dirichlet(np.ones(self.num_cells))
                signatures[region_id][channel_id] = probs
        return signatures

    def _sample_observation(self, true_region: int, channel_id: int) -> int:
        """Sample an observation from the true region's signature."""
        probs = self.signatures[true_region][channel_id]
        return np.random.choice(len(probs), p=probs)

    def _create_complement_mapping(self) -> Dict:
        """Create a deterministic, involutive complement mapping."""
        # For each channel, create an inverse channel
        # For each cell in a channel, create an inverse cell
        mapping = {}
        for c in range(self.num_channels):
            mapping[f'c_{c}'] = f'c_{self.num_channels - 1 - c}'
        for v in range(self.num_cells):
            mapping[f'v_{v}'] = f'v_{self.num_cells - 1 - v}'
        return mapping

    def _apply_complement(self, evidence: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Apply complement operation to evidence stream."""
        complement = []
        for channel_id, cell_id in evidence:
            # Inverse channel: mirror around center
            comp_channel = self.num_channels - 1 - channel_id
            # Inverse cell: mirror around center
            comp_cell = self.num_cells - 1 - cell_id
            complement.append((comp_channel, comp_cell))
        return complement

    def _is_involutive(self, evidence: List[Tuple[int, int]]) -> bool:
        """Verify that complement is involutive: τ̄̄ = τ"""
        comp_once = self._apply_complement(evidence)
        comp_twice = self._apply_complement(comp_once)
        return comp_twice == evidence

    def experiment_complement_reconstruction(self, num_trials: int = 100) -> Dict:
        """Verify that B can reconstruct message from complement."""
        print("Running: Complement Reconstruction Experiment")

        results = {
            'trials': []
        }

        for trial in range(num_trials):
            true_region = np.random.randint(0, self.num_regions)

            # Generate evidence
            evidence = []
            for _ in range(15):
                channel_id = np.random.randint(0, self.num_channels)
                cell_id = self._sample_observation(true_region, channel_id)
                evidence.append((channel_id, cell_id))

            # Compute complement
            complement = self._apply_complement(evidence)

            # B observes complement and reconstructs original
            reconstructed = self._apply_complement(complement)

            # Check reconstruction accuracy
            reconstruction_exact = reconstructed == evidence
            involution_holds = self._is_involutive(evidence)

            results['trials'].append({
                'reconstruction_exact': bool(reconstruction_exact),
                'involution_holds': bool(involution_holds),
                'evidence_length': len(evidence)
            })

        results['summary'] = {
            'reconstruction_accuracy': float(np.mean([t['reconstruction_exact'] for t in results['trials']])),
            'involution_rate': float(np.mean([t['involution_holds'] for t in results['trials']])),
            'total_trials': num_trials
        }

        self.results['complement_reconstruction'] = results
        return results

    def experiment_complement_consistency(self, num_trials: int = 100) -> Dict:
        """Verify that τ and τ̄ satisfy complementarity relation."""
        print("Running: Complement Consistency Experiment")

        results = {
            'trials': []
        }

        for trial in range(num_trials):
            true_region = np.random.randint(0, self.num_regions)

            # Generate evidence
            evidence = []
            for _ in range(15):
                channel_id = np.random.randint(0, self.num_channels)
                cell_id = self._sample_observation(true_region, channel_id)
                evidence.append((channel_id, cell_id))

            complement = self._apply_complement(evidence)

            # Compute likelihood of τ
            tau_likelihood = 1.0
            for channel_id, cell_id in evidence:
                tau_likelihood *= self.signatures[true_region][channel_id][cell_id]

            # Compute likelihood of τ̄
            comp_likelihood = 1.0
            for channel_id, cell_id in complement:
                # Complement comes from inverse signature
                comp_region = true_region  # Same region, just inverse evidence
                comp_likelihood *= self.signatures[comp_region][channel_id][cell_id]

            # Consistency: both should have non-zero likelihood
            tau_valid = tau_likelihood > 1e-10
            comp_valid = comp_likelihood > 1e-10

            # Verify involution
            involution_holds = self._is_involutive(evidence)

            results['trials'].append({
                'tau_likelihood': float(tau_likelihood),
                'complement_likelihood': float(comp_likelihood),
                'tau_valid': bool(tau_valid),
                'complement_valid': bool(comp_valid),
                'involution_holds': bool(involution_holds)
            })

        results['summary'] = {
            'tau_validity_rate': float(np.mean([t['tau_valid'] for t in results['trials']])),
            'complement_validity_rate': float(np.mean([t['complement_valid'] for t in results['trials']])),
            'both_valid_rate': float(np.mean([t['tau_valid'] and t['complement_valid'] for t in results['trials']])),
            'involution_rate': float(np.mean([t['involution_holds'] for t in results['trials']]))
        }

        self.results['complement_consistency'] = results
        return results

    def experiment_forgery_amplification(self, num_trials: int = 100) -> Dict:
        """Verify that forging both τ and τ̄ consistently is harder than forging τ alone."""
        print("Running: Forgery Amplification Experiment")

        results = {
            'trials': []
        }

        for trial in range(num_trials):
            true_region = np.random.randint(0, self.num_regions)
            attacker_region = (true_region + 2) % self.num_regions

            # Generate true evidence from true region
            true_evidence = []
            for _ in range(15):
                channel_id = np.random.randint(0, self.num_channels)
                cell_id = self._sample_observation(true_region, channel_id)
                true_evidence.append((channel_id, cell_id))

            # Attacker forges evidence from their region
            forged_evidence = []
            for _ in range(15):
                channel_id = np.random.randint(0, self.num_channels)
                cell_id = self._sample_observation(attacker_region, channel_id)
                forged_evidence.append((channel_id, cell_id))

            true_complement = self._apply_complement(true_evidence)
            forged_complement = self._apply_complement(forged_evidence)

            # Compute likelihoods for true evidence
            true_tau_likelihood = np.prod([self.signatures[true_region][c][v]
                                          for c, v in true_evidence])
            true_comp_likelihood = np.prod([self.signatures[true_region][c][v]
                                           for c, v in true_complement])

            # Compute likelihoods for forged evidence
            forged_tau_likelihood = np.prod([self.signatures[attacker_region][c][v]
                                            for c, v in forged_evidence])
            forged_comp_likelihood = np.prod([self.signatures[attacker_region][c][v]
                                             for c, v in forged_complement])

            # Detection metrics
            # PLM detection: can we distinguish true τ from forged τ?
            plm_detection_ratio = np.log((true_tau_likelihood + 1e-20) / (forged_tau_likelihood + 1e-20))

            # CSPLM detection: must forge both τ and τ̄ consistently
            # Attacker must forge τ such that it produces a valid complement
            csplm_detection_ratio = np.log((true_tau_likelihood * true_comp_likelihood + 1e-20) /
                                          (forged_tau_likelihood * forged_comp_likelihood + 1e-20))

            # Amplification factor: how much harder is CSPLM than PLM?
            amplification = abs(csplm_detection_ratio) / (abs(plm_detection_ratio) + 1e-10)

            results['trials'].append({
                'plm_detection_ratio': float(plm_detection_ratio),
                'csplm_detection_ratio': float(csplm_detection_ratio),
                'amplification_factor': float(amplification),
                'csplm_harder': bool(abs(csplm_detection_ratio) > abs(plm_detection_ratio))
            })

        results['summary'] = {
            'mean_plm_detection_ratio': float(np.mean([abs(t['plm_detection_ratio']) for t in results['trials']])),
            'mean_csplm_detection_ratio': float(np.mean([abs(t['csplm_detection_ratio']) for t in results['trials']])),
            'mean_amplification_factor': float(np.mean([t['amplification_factor'] for t in results['trials']])),
            'csplm_harder_rate': float(np.mean([t['csplm_harder'] for t in results['trials']]))
        }

        self.results['forgery_amplification'] = results
        return results

    def experiment_complement_noise_detection(self, num_trials: int = 100) -> Dict:
        """Verify that corruption of complement is detectable."""
        print("Running: Complement Noise Detection Experiment")

        results = {
            'corruption_rates': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            'trials': []
        }

        for trial in range(num_trials):
            true_region = np.random.randint(0, self.num_regions)

            # Generate true evidence
            evidence = []
            for _ in range(15):
                channel_id = np.random.randint(0, self.num_channels)
                cell_id = self._sample_observation(true_region, channel_id)
                evidence.append((channel_id, cell_id))

            true_complement = self._apply_complement(evidence)

            trial_results = []

            for corruption_rate in results['corruption_rates']:
                # Corrupt the complement
                corrupted_complement = list(true_complement)
                num_corruptions = int(len(corrupted_complement) * corruption_rate)

                for _ in range(num_corruptions):
                    idx = np.random.randint(0, len(corrupted_complement))
                    channel_id, cell_id = corrupted_complement[idx]
                    # Flip to a random cell
                    new_cell_id = np.random.randint(0, self.num_cells)
                    corrupted_complement[idx] = (channel_id, new_cell_id)

                corrupted_complement = tuple(corrupted_complement)

                # Try to reconstruct and verify
                reconstructed = self._apply_complement(corrupted_complement)
                reconstruction_matches = reconstructed == evidence
                involution_holds = self._is_involutive(evidence) if corruption_rate == 0.0 else False

                # Compute consistency likelihood
                consistency_likelihood = 1.0
                for channel_id, cell_id in corrupted_complement:
                    consistency_likelihood *= self.signatures[true_region][channel_id][cell_id]

                # True complement likelihood
                true_likelihood = 1.0
                for channel_id, cell_id in true_complement:
                    true_likelihood *= self.signatures[true_region][channel_id][cell_id]

                detected = consistency_likelihood < (true_likelihood * 0.9)

                trial_results.append({
                    'corruption_rate': float(corruption_rate),
                    'reconstruction_matches': bool(reconstruction_matches),
                    'consistency_likelihood': float(consistency_likelihood),
                    'true_likelihood': float(true_likelihood),
                    'detected': bool(detected)
                })

            results['trials'].append(trial_results)

        # Summary by corruption rate
        results['summary'] = {}
        for i, corruption_rate in enumerate(results['corruption_rates']):
            detections = [trial[i]['detected'] for trial in results['trials']
                         if i < len(trial)]
            results['summary'][f'corruption_{int(corruption_rate*100)}%'] = {
                'detection_rate': float(np.mean(detections)) if detections else 0.0
            }

        self.results['complement_noise_detection'] = results
        return results

    def experiment_csplm_vs_plm(self, num_trials: int = 100) -> Dict:
        """Compare CSPLM security to PLM baseline."""
        print("Running: CSPLM vs PLM Comparison Experiment")

        results = {
            'trials': []
        }

        for trial in range(num_trials):
            true_region = np.random.randint(0, self.num_regions)
            attacker_region = (true_region + 1) % self.num_regions

            # Generate true evidence
            true_evidence = []
            for _ in range(20):
                channel_id = np.random.randint(0, self.num_channels)
                cell_id = self._sample_observation(true_region, channel_id)
                true_evidence.append((channel_id, cell_id))

            # Attacker generates forged evidence
            forged_evidence = []
            for _ in range(20):
                channel_id = np.random.randint(0, self.num_channels)
                cell_id = self._sample_observation(attacker_region, channel_id)
                forged_evidence.append((channel_id, cell_id))

            # PLM: Compute posterior margin (state change detection)
            true_posterior_likelihood = np.prod([self.signatures[true_region][c][v]
                                                for c, v in true_evidence])
            forged_posterior_likelihood = np.prod([self.signatures[attacker_region][c][v]
                                                 for c, v in forged_evidence])

            plm_margin = abs(np.log(true_posterior_likelihood / (forged_posterior_likelihood + 1e-20)))

            # CSPLM: Compute margin for both τ and τ̄
            true_complement = self._apply_complement(true_evidence)
            forged_complement = self._apply_complement(forged_evidence)

            true_comp_likelihood = np.prod([self.signatures[true_region][c][v]
                                           for c, v in true_complement])
            forged_comp_likelihood = np.prod([self.signatures[attacker_region][c][v]
                                             for c, v in forged_complement])

            csplm_margin = abs(np.log((true_posterior_likelihood * true_comp_likelihood) /
                                     ((forged_posterior_likelihood * forged_comp_likelihood) + 1e-20)))

            results['trials'].append({
                'plm_detection_margin': float(plm_margin),
                'csplm_detection_margin': float(csplm_margin),
                'csplm_advantage': float(csplm_margin / (plm_margin + 1e-10)),
                'csplm_superior': bool(csplm_margin > plm_margin)
            })

        results['summary'] = {
            'mean_plm_margin': float(np.mean([t['plm_detection_margin'] for t in results['trials']])),
            'mean_csplm_margin': float(np.mean([t['csplm_detection_margin'] for t in results['trials']])),
            'mean_advantage_factor': float(np.mean([t['csplm_advantage'] for t in results['trials']])),
            'csplm_superior_rate': float(np.mean([t['csplm_superior'] for t in results['trials']]))
        }

        self.results['csplm_vs_plm'] = results
        return results

    def run_all_experiments(self) -> Dict:
        """Run all CSPLM validation experiments."""
        print("\n" + "="*60)
        print("CSPLM VALIDATION EXPERIMENTS")
        print("="*60 + "\n")

        self.experiment_complement_reconstruction()
        self.experiment_complement_consistency()
        self.experiment_forgery_amplification()
        self.experiment_complement_noise_detection()
        self.experiment_csplm_vs_plm()

        self.results['metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'num_regions': self.num_regions,
            'num_channels': self.num_channels,
            'num_cells_per_channel': self.num_cells
        }

        print("\n" + "="*60)
        print("ALL CSPLM EXPERIMENTS COMPLETED")
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
    validator = CSPLMValidator(num_regions=9, num_channels=3, num_cells_per_channel=3)
    results = validator.run_all_experiments()
    validator.save_results('csplm_validation_results.json')
