"""
Assessment Configuration Loader
================================

Loads domain-agnostic assessment settings from config/assessment.yaml
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml


def get_config_path() -> Path:
    """Get path to assessment config file."""
    # Try relative to prism package
    prism_root = Path(__file__).parent.parent.parent
    config_path = prism_root / 'config' / 'assessment.yaml'

    if config_path.exists():
        return config_path

    # Try current working directory
    cwd_path = Path.cwd() / 'config' / 'assessment.yaml'
    if cwd_path.exists():
        return cwd_path

    raise FileNotFoundError(
        f"Assessment config not found at {config_path} or {cwd_path}"
    )


def load_assessment_config() -> Dict[str, Any]:
    """Load the full assessment configuration."""
    config_path = get_config_path()
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_domain_config(domain: str) -> Dict[str, Any]:
    """
    Get merged configuration for a specific domain.

    Returns defaults with domain-specific overrides applied.
    """
    config = load_assessment_config()
    defaults = config.get('defaults', {})
    domains = config.get('domains', {})

    if domain not in domains:
        print(f"Warning: Domain '{domain}' not in config, using defaults")
        return defaults

    domain_config = domains[domain]

    # Deep merge: domain overrides defaults
    merged = _deep_merge(defaults, domain_config)
    return merged


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Deep merge two dictionaries, override takes precedence."""
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value

    return result


# Convenience accessors

def get_windows(domain: str) -> Dict[str, int]:
    """Get window configuration for domain."""
    config = get_domain_config(domain)
    return config.get('windows', {
        'pre_onset': 7,
        'post_onset': 5,
        'mode1_lookback': 7,
    })


def get_thresholds(domain: str) -> Dict[str, float]:
    """Get detection thresholds for domain."""
    config = get_domain_config(domain)
    return config.get('thresholds', {
        'break_zscore': 1.0,
        'affinity_drop': 0.1,
        'mode_entropy_spike': 0.5,
    })


def get_what_features(domain: str) -> List[str]:
    """Get WHAT layer features for domain."""
    config = get_domain_config(domain)
    return config.get('what_features', [
        'alpha', 'beta', 'omega', 'unconditional_vol',
        'spectral_slope', 'spectral_entropy',
        'permutation_entropy', 'sample_entropy',
    ])


def get_when_features(domain: str) -> List[str]:
    """Get WHEN layer features for domain."""
    config = get_domain_config(domain)
    return config.get('when_features', [
        'break_n', 'break_rate', 'break_is_accelerating',
        'dirac_n_impulses', 'dirac_mean_magnitude',
        'heaviside_n_steps', 'heaviside_mean_magnitude',
    ])


def get_mode_features(domain: str) -> List[str]:
    """Get MODE layer features for domain."""
    config = get_domain_config(domain)
    return config.get('mode_features', [
        'gradient_mean', 'gradient_std', 'gradient_magnitude',
        'laplacian_mean', 'laplacian_std', 'divergence',
    ])


def get_signal_patterns(domain: str) -> Dict[str, str]:
    """Get signal naming patterns for domain."""
    config = get_domain_config(domain)
    return {
        'prefix': config.get('signal_prefix', ''),
        'fault_signal': config.get('fault_signal', ''),
        'exclude_pattern': config.get('exclude_pattern', ''),
    }


def get_precursor_mode(domain: str) -> int:
    """Get the precursor mode ID for domain."""
    config = get_domain_config(domain)
    clustering = config.get('mode_clustering', {})
    return clustering.get('precursor_mode', 1)


def get_precursor_signals(domain: str) -> List[str]:
    """Get known precursor signals for domain."""
    config = get_domain_config(domain)
    return config.get('precursor_signals', [])


def print_config(domain: str):
    """Print the effective configuration for a domain."""
    config = get_domain_config(domain)

    print(f"Assessment Configuration for: {domain}")
    print("=" * 60)
    print(f"Description: {config.get('description', 'N/A')}")
    print()

    print("Signal Patterns:")
    print(f"  Prefix:          {config.get('signal_prefix', 'N/A')}")
    print(f"  Fault Signal: {config.get('fault_signal', 'N/A')}")
    print(f"  Exclude Pattern: {config.get('exclude_pattern', 'N/A')}")
    print()

    windows = config.get('windows', {})
    print("Windows:")
    print(f"  Pre-onset:       {windows.get('pre_onset', 7)} days")
    print(f"  Post-onset:      {windows.get('post_onset', 5)} days")
    print(f"  Mode1 lookback:  {windows.get('mode1_lookback', 7)} days")
    print()

    thresholds = config.get('thresholds', {})
    print("Thresholds:")
    print(f"  Break z-score:   {thresholds.get('break_zscore', 1.0)}")
    print(f"  Affinity drop:   {thresholds.get('affinity_drop', 0.1)}")
    print()

    print(f"WHAT features: {len(config.get('what_features', []))}")
    print(f"WHEN features: {len(config.get('when_features', []))}")
    print(f"MODE features: {len(config.get('mode_features', []))}")
    print()


if __name__ == '__main__':
    import sys
    domain = sys.argv[1] if len(sys.argv) > 1 else 'cheme'
    print_config(domain)
