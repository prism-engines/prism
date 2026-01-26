"""
PRISM Domain Engines
====================

Domain-specific engines that require field-specific knowledge.
Loaded when config specifies a domain.

Available Domains:
    - prism: PRISM-specific degradation model (barycenter, modes, systems)
    - battery: Battery diagnostics (SOC, SOH, capacity)
    - fluid: Fluid mechanics (Reynolds, pressure drop)
    - heat_transfer: Heat conduction/convection (Fourier)
    - mass_transfer: Diffusion (Fick)
    - thermodynamics: Gibbs energy, EOS
    - chemical: Reaction kinetics (Arrhenius, CSTR)
    - control: Process control (PID, transfer functions)
    - separations: Unit operations (distillation, extraction)
    - engineering: Dimensionless numbers
    - diagnostics: Degradation-focused wavelet

Usage:
    In config.yaml:
        domain: battery  # loads domains/battery/ engines

    In orchestrator:
        if config.get('domain'):
            domain_engines = load_engines(f'prism/engines/domains/{domain}/')
"""

AVAILABLE_DOMAINS = [
    'prism',
    'battery',
    'fluid',
    'heat_transfer',
    'mass_transfer',
    'thermodynamics',
    'chemical',
    'control',
    'separations',
    'engineering',
    'diagnostics',
]

__all__ = ['AVAILABLE_DOMAINS']
