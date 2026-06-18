//! Similarity / distance metrics.

/// A vector comparison metric.
///
/// For [`Metric::Cosine`] and [`Metric::DotProduct`] a *larger* value means
/// *more similar*; for [`Metric::L2`] a *smaller* value means *more similar*.
/// The retrieval kernels normalise this difference internally so that a single
/// "higher is better" ranking key drives the top-k selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Metric {
    /// Cosine similarity, `⟨a, b⟩ / (‖a‖ · ‖b‖)`, in `[-1, 1]`.
    Cosine,
    /// Raw inner product `⟨a, b⟩`.
    DotProduct,
    /// Euclidean (L2) distance `‖a − b‖₂`, in `[0, ∞)`.
    L2,
}

impl Metric {
    /// Returns `true` when larger metric values indicate greater similarity.
    #[inline]
    #[must_use]
    pub const fn higher_is_better(self) -> bool {
        matches!(self, Metric::Cosine | Metric::DotProduct)
    }

    /// The canonical lowercase name of the metric.
    #[inline]
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Metric::Cosine => "cosine",
            Metric::DotProduct => "dot",
            Metric::L2 => "l2",
        }
    }

    /// Parses a metric from a case-insensitive name, accepting common aliases.
    ///
    /// Returns `None` for unrecognised names so the caller can produce a
    /// domain-appropriate error (e.g. a Python `ValueError`).
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "cosine" | "cos" => Some(Metric::Cosine),
            "dot" | "dot_product" | "ip" | "inner_product" => Some(Metric::DotProduct),
            "l2" | "euclidean" | "euclid" => Some(Metric::L2),
            _ => None,
        }
    }
}
