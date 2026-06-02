"""Statistics dataclasses collected during MountainSort5 sorting pipelines.

These are pure data containers.  They are populated *observationally* by the
sorting functions and serialised to CSV by the caller (TianSuo).  No algorithm
logic lives here.
"""

from dataclasses import dataclass, field


@dataclass
class Scheme1SortingStats:
    """Intermediate statistics collected during :func:`sorting_scheme1`.

    Every field is optional (defaults to 0 / empty) so that the sorting
    function can populate only what is available without worrying about
    partial runs or early exits.
    """

    # -- Recording ----------------------------------------------------------
    num_channels: int = 0
    num_frames: int = 0
    sampling_frequency: float = 0.0

    # -- Spike detection ----------------------------------------------------
    num_spikes_detected: int = 0

    # -- After deduplication ------------------------------------------------
    num_spikes_after_dedup: int = 0

    # -- First clustering pass ----------------------------------------------
    num_clusters_before_alignment: int = 0

    # -- Template alignment -------------------------------------------------
    alignment_performed: bool = False
    alignment_iterations: int = 0
    alignment_offsets_mean: float = 0.0
    alignment_offsets_std: float = 0.0

    # -- Second clustering pass (after alignment, if performed) -------------
    num_clusters_after_alignment: int = 0

    # -- Out-of-bounds filtering --------------------------------------------
    num_spikes_oob_removed: int = 0

    # -- Final results ------------------------------------------------------
    num_spikes_final: int = 0
    num_units_final: int = 0

    # -- Per-unit metrics (from SortingSchemeExtraOutput) -------------------
    unit_spike_counts: list = field(default_factory=list)
    unit_peak_channels: list = field(default_factory=list)
    unit_template_amplitudes: list = field(default_factory=list)


@dataclass
class Scheme2SortingStats:
    """Intermediate statistics collected during :func:`sorting_scheme2`.

    Phase 1 runs on a (possibly subsampled) training recording, therefore
    its stats are nested inside ``phase1`` to distinguish them from the
    full-recording phase 2 statistics.
    """

    # -- Phase 1 (on training recording) ------------------------------------
    phase1: Scheme1SortingStats = field(default_factory=Scheme1SortingStats)

    # -- Training -----------------------------------------------------------
    training_duration_actual_sec: float = 0.0
    num_training_units: int = 0

    # -- Phase 2 chunk processing (aggregated across all chunks) ------------
    num_chunks: int = 0
    num_spikes_detected_phase2: int = 0
    num_spikes_after_classification: int = 0
    num_spikes_after_dedup_phase2: int = 0

    # -- Final results ------------------------------------------------------
    num_spikes_final: int = 0
    num_units_final: int = 0

    # -- Per-unit metrics (from final sorting) ------------------------------
    unit_spike_counts: list = field(default_factory=list)
