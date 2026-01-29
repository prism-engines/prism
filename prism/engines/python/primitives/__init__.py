"""
PRISM Primitives Library

116 atomic functions organized by type:

Individual (1-35): Single-signal computations
- statistics: mean, std, variance, min_max, percentiles, skewness, kurtosis, rms, peak_to_peak, crest_factor, zero_crossings, mean_crossings
- calculus: derivative, integral, curvature
- correlation: autocorrelation, partial_autocorrelation
- spectral: fft, psd, dominant_frequency, spectral_centroid, spectral_bandwidth, spectral_entropy, wavelet_coeffs
- hilbert: envelope, hilbert_transform, instantaneous_frequency, instantaneous_amplitude
- entropy: sample_entropy, permutation_entropy, approximate_entropy
- fractal: hurst_exponent, dfa, hurst_r2
- stationarity: stationarity_test, trend, changepoints, mann_kendall_test

Pairwise (36-55): Two-signal computations
- correlation: correlation, covariance, cross_correlation, lag_at_max_xcorr, partial_correlation
- spectral: coherence, cross_spectral_density, phase_spectrum, wavelet_coherence
- information: mutual_information, transfer_entropy
- causality: granger_causality, convergent_cross_mapping
- distance: dynamic_time_warping, euclidean_distance, cosine_similarity, manhattan_distance
- regression: linear_regression, ratio, product, difference, sum_signals

Matrix (56-65): All-signals computations
- covariance: covariance_matrix, correlation_matrix
- decomposition: eigendecomposition, svd, pca_loadings, factor_scores
- dmd: dynamic_mode_decomposition
- graph: distance_matrix, adjacency_matrix, laplacian_matrix

Embedding (66-69): Phase space reconstruction
- delay: time_delay_embedding, optimal_delay, optimal_dimension, multivariate_embedding

Topology (70-74): Persistent homology
- persistence: persistence_diagram, betti_numbers, persistence_entropy, persistence_landscape
- distance: wasserstein_distance, bottleneck_distance

Network (75-85): Graph metrics
- structure: threshold_matrix, network_density, clustering_coefficient, connected_components, assortativity
- centrality: centrality_degree, centrality_betweenness, centrality_eigenvector, centrality_closeness
- paths: shortest_paths, average_path_length, diameter
- community: modularity, community_detection

Dynamical (86-95): Chaos and recurrence
- lyapunov: lyapunov_rosenstein, lyapunov_kantz, lyapunov_spectrum
- dimension: correlation_dimension, correlation_integral, information_dimension
- rqa: recurrence_matrix, recurrence_rate, determinism, laminarity, trapping_time, entropy_rqa, max_diagonal_line, divergence_rqa

Tests (96-107): Statistical tests
- hypothesis: t_test, t_test_paired, t_test_independent, f_test, chi_squared_test, mannwhitney_test, kruskal_test, anova
- normalization: z_score, min_max_scale, robust_scale
- stationarity_tests: adf_test, kpss_test, philips_perron_test
- bootstrap: bootstrap_ci, bootstrap_mean, bootstrap_std, permutation_test

Information (108-116): Information theory
- entropy: shannon_entropy, renyi_entropy, tsallis_entropy, joint_entropy, conditional_entropy
- divergence: cross_entropy, kl_divergence, js_divergence
- mutual: mutual_information, conditional_mutual_information, multivariate_mutual_information, total_correlation, interaction_information
"""

# Individual primitives (1-35)
from .individual import (
    # Statistics (1-12)
    mean, std, variance, min_max, percentiles,
    skewness, kurtosis, rms, peak_to_peak,
    crest_factor, zero_crossings, mean_crossings,
    # Calculus (13-14)
    derivative, integral, curvature,
    # Correlation (15-16)
    autocorrelation, partial_autocorrelation,
    # Spectral (17-23)
    fft, psd, dominant_frequency, spectral_centroid,
    spectral_bandwidth, spectral_entropy, wavelet_coeffs,
    # Hilbert (24-27)
    envelope, hilbert_transform, instantaneous_frequency, instantaneous_amplitude,
    # Entropy (28-30)
    sample_entropy, permutation_entropy, approximate_entropy,
    # Fractal (31-32)
    hurst_exponent, dfa, hurst_r2,
    # Stationarity (33-35)
    stationarity_test, trend, changepoints, mann_kendall_test,
)

# Pairwise primitives (36-55)
from .pairwise import (
    # Correlation (36-40, 47)
    correlation, covariance, cross_correlation, lag_at_max_xcorr, partial_correlation,
    # Spectral (40-43)
    coherence, cross_spectral_density, phase_spectrum, wavelet_coherence,
    # Information (44-45)
    mutual_information as pairwise_mutual_information,
    transfer_entropy,
    # Causality (46, 48)
    granger_causality, convergent_cross_mapping,
    # Distance (49-51)
    dynamic_time_warping, euclidean_distance, cosine_similarity, manhattan_distance,
    # Regression (52-55)
    linear_regression, ratio, product, difference, sum_signals,
)

# Matrix primitives (56-65)
from .matrix import (
    # Covariance (56-57)
    covariance_matrix, correlation_matrix,
    # Decomposition (58-62)
    eigendecomposition, svd, pca_loadings, factor_scores,
    # DMD (60)
    dynamic_mode_decomposition,
    # Graph (63-65)
    distance_matrix, adjacency_matrix, laplacian_matrix,
)

# Embedding primitives (66-69)
from .embedding import (
    time_delay_embedding, optimal_delay, optimal_dimension, multivariate_embedding,
)

# Topology primitives (70-74)
from .topology import (
    persistence_diagram, betti_numbers, persistence_entropy, persistence_landscape,
    wasserstein_distance, bottleneck_distance,
)

# Network primitives (75-85)
from .network import (
    # Structure (75-77, 84)
    threshold_matrix, network_density, clustering_coefficient, connected_components, assortativity,
    # Centrality (79-82)
    centrality_degree, centrality_betweenness, centrality_eigenvector, centrality_closeness,
    # Paths (83)
    shortest_paths, average_path_length, diameter,
    # Community (78, 85)
    modularity, community_detection,
)

# Dynamical primitives (86-95)
from .dynamical import (
    # Lyapunov (86-87)
    lyapunov_rosenstein, lyapunov_kantz, lyapunov_spectrum,
    # Dimension (88-89)
    correlation_dimension, correlation_integral, information_dimension,
    kaplan_yorke_dimension,
    # RQA (90-95)
    recurrence_matrix, recurrence_rate, determinism, laminarity,
    trapping_time, entropy_rqa, max_diagonal_line, divergence_rqa,
)

# Test primitives (96-107)
from .tests import (
    # Hypothesis (96, 99, 102-103)
    t_test, t_test_paired, t_test_independent, f_test,
    chi_squared_test, mannwhitney_test, kruskal_test, anova,
    shapiro_test, levene_test,
    # Nonparametric (98)
    mann_kendall,
    # Normalization (97)
    z_score, z_score_significance, min_max_scale, robust_scale,
    # Stationarity (100-101)
    adf_test, kpss_test, philips_perron_test,
    # Bootstrap (104-105)
    bootstrap_ci, bootstrap_mean, bootstrap_std, permutation_test,
    # Null models (106-107)
    surrogate_test, marchenko_pastur_test, significance_summary,
)

# Information primitives (108-116)
from .information import (
    # Entropy (108-110)
    shannon_entropy, renyi_entropy, tsallis_entropy,
    joint_entropy, conditional_entropy,
    # Mutual information (111-112)
    mutual_information, conditional_mutual_information,
    multivariate_mutual_information, total_correlation, interaction_information,
    # Transfer entropy (113)
    transfer_entropy,
    # Partial information decomposition (114-116)
    partial_information_decomposition, redundancy, synergy,
    # Divergences (additional)
    cross_entropy, kl_divergence, js_divergence,
)

__all__ = [
    # Individual (1-35)
    'mean', 'std', 'variance', 'min_max', 'percentiles',
    'skewness', 'kurtosis', 'rms', 'peak_to_peak',
    'crest_factor', 'zero_crossings', 'mean_crossings',
    'derivative', 'integral', 'curvature',
    'autocorrelation', 'partial_autocorrelation',
    'fft', 'psd', 'dominant_frequency', 'spectral_centroid',
    'spectral_bandwidth', 'spectral_entropy', 'wavelet_coeffs',
    'envelope', 'hilbert_transform', 'instantaneous_frequency', 'instantaneous_amplitude',
    'sample_entropy', 'permutation_entropy', 'approximate_entropy',
    'hurst_exponent', 'dfa', 'hurst_r2',
    'stationarity_test', 'trend', 'changepoints', 'mann_kendall_test',
    # Pairwise (36-55)
    'correlation', 'covariance', 'cross_correlation', 'lag_at_max_xcorr', 'partial_correlation',
    'coherence', 'cross_spectral_density', 'phase_spectrum', 'wavelet_coherence',
    'pairwise_mutual_information', 'transfer_entropy',
    'granger_causality', 'convergent_cross_mapping',
    'dynamic_time_warping', 'euclidean_distance', 'cosine_similarity', 'manhattan_distance',
    'linear_regression', 'ratio', 'product', 'difference', 'sum_signals',
    # Matrix (56-65)
    'covariance_matrix', 'correlation_matrix',
    'eigendecomposition', 'svd', 'pca_loadings', 'factor_scores',
    'dynamic_mode_decomposition',
    'distance_matrix', 'adjacency_matrix', 'laplacian_matrix',
    # Embedding (66-69)
    'time_delay_embedding', 'optimal_delay', 'optimal_dimension', 'multivariate_embedding',
    # Topology (70-74)
    'persistence_diagram', 'betti_numbers', 'persistence_entropy', 'persistence_landscape',
    'wasserstein_distance', 'bottleneck_distance',
    # Network (75-85)
    'threshold_matrix', 'network_density', 'clustering_coefficient', 'connected_components', 'assortativity',
    'centrality_degree', 'centrality_betweenness', 'centrality_eigenvector', 'centrality_closeness',
    'shortest_paths', 'average_path_length', 'diameter',
    'modularity', 'community_detection',
    # Dynamical (86-95)
    'lyapunov_rosenstein', 'lyapunov_kantz', 'lyapunov_spectrum',
    'correlation_dimension', 'correlation_integral', 'information_dimension',
    'kaplan_yorke_dimension',
    'recurrence_matrix', 'recurrence_rate', 'determinism', 'laminarity',
    'trapping_time', 'entropy_rqa', 'max_diagonal_line', 'divergence_rqa',
    # Tests (96-107)
    't_test', 't_test_paired', 't_test_independent', 'f_test',
    'chi_squared_test', 'mannwhitney_test', 'kruskal_test', 'anova',
    'shapiro_test', 'levene_test',
    'mann_kendall',
    'z_score', 'z_score_significance', 'min_max_scale', 'robust_scale',
    'adf_test', 'kpss_test', 'philips_perron_test',
    'bootstrap_ci', 'bootstrap_mean', 'bootstrap_std', 'permutation_test',
    'surrogate_test', 'marchenko_pastur_test', 'significance_summary',
    # Information (108-116)
    'shannon_entropy', 'renyi_entropy', 'tsallis_entropy',
    'joint_entropy', 'conditional_entropy',
    'mutual_information', 'conditional_mutual_information',
    'multivariate_mutual_information', 'total_correlation', 'interaction_information',
    'transfer_entropy',
    'partial_information_decomposition', 'redundancy', 'synergy',
    'cross_entropy', 'kl_divergence', 'js_divergence',
]
