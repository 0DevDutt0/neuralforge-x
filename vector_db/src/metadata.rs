//! Per-vector metadata and the filter predicate language.
//!
//! Every stored vector carries an optional bag of scalar metadata. Searches may
//! supply a [`Filter`] that is evaluated against that metadata *during* graph
//! traversal: non-matching nodes are still used as routing waypoints (so the
//! graph stays connected) but are never admitted to the result set. This is the
//! standard "filtered HNSW" trade-off — correctness is preserved, and recall
//! degrades gracefully as the predicate gets more selective.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// A scalar metadata value.
///
/// The set is deliberately small — the kinds that survive a round-trip through
/// JSON and a Parquet string column — so the store never has to reason about
/// nested documents.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MetaValue {
    /// An explicit null / missing value.
    Null,
    /// A boolean.
    Bool(bool),
    /// A 64-bit signed integer.
    Int(i64),
    /// A 64-bit float.
    Float(f64),
    /// A UTF-8 string.
    Str(String),
}

impl MetaValue {
    /// Interprets the value as an `f64` for numeric comparison, if possible.
    ///
    /// `Int` and `Float` convert directly; `Bool` maps to `0.0`/`1.0`; strings
    /// and nulls are not numeric and yield `None`.
    #[must_use]
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            MetaValue::Int(i) => Some(*i as f64),
            MetaValue::Float(f) => Some(*f),
            MetaValue::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
            _ => None,
        }
    }
}

impl From<&str> for MetaValue {
    fn from(s: &str) -> Self {
        MetaValue::Str(s.to_owned())
    }
}
impl From<String> for MetaValue {
    fn from(s: String) -> Self {
        MetaValue::Str(s)
    }
}
impl From<i64> for MetaValue {
    fn from(i: i64) -> Self {
        MetaValue::Int(i)
    }
}
impl From<f64> for MetaValue {
    fn from(f: f64) -> Self {
        MetaValue::Float(f)
    }
}
impl From<bool> for MetaValue {
    fn from(b: bool) -> Self {
        MetaValue::Bool(b)
    }
}

/// A vector's metadata: an ordered map of field name to scalar value.
///
/// `BTreeMap` (rather than `HashMap`) keeps the field order deterministic, so a
/// snapshot serialises identically across runs.
pub type Metadata = BTreeMap<String, MetaValue>;

/// A boolean predicate over a vector's [`Metadata`].
///
/// Filters compose: [`Filter::all`]/[`Filter::any`] form conjunctions and
/// disjunctions, and [`Filter::not`] negates. A field that is absent compares as
/// "no match" for every leaf except [`Filter::Missing`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Filter {
    /// Matches every vector. The default when no filter is supplied.
    Any,
    /// The field exists and equals the given value.
    Eq(String, MetaValue),
    /// The field is absent, or exists and differs from the given value.
    Ne(String, MetaValue),
    /// The field is numeric and strictly less than the bound.
    Lt(String, f64),
    /// The field is numeric and less than or equal to the bound.
    Le(String, f64),
    /// The field is numeric and strictly greater than the bound.
    Gt(String, f64),
    /// The field is numeric and greater than or equal to the bound.
    Ge(String, f64),
    /// The field exists and is one of the listed values.
    In(String, Vec<MetaValue>),
    /// The field is present (with any non-null value).
    Exists(String),
    /// The field is absent or null.
    Missing(String),
    /// Logical conjunction — all sub-filters must match.
    And(Vec<Filter>),
    /// Logical disjunction — at least one sub-filter must match.
    Or(Vec<Filter>),
    /// Logical negation.
    Not(Box<Filter>),
}

impl Filter {
    /// A conjunction of the given filters (`true` for an empty list).
    #[must_use]
    pub fn all(filters: impl IntoIterator<Item = Filter>) -> Self {
        Filter::And(filters.into_iter().collect())
    }

    /// A disjunction of the given filters (`false` for an empty list).
    #[must_use]
    pub fn any(filters: impl IntoIterator<Item = Filter>) -> Self {
        Filter::Or(filters.into_iter().collect())
    }

    /// Negates this filter. (Named `negate` rather than `not` to avoid colliding
    /// with [`std::ops::Not`]; the `!` operator is not implemented for filters.)
    #[must_use]
    pub fn negate(self) -> Self {
        Filter::Not(Box::new(self))
    }

    /// Evaluates the predicate against a vector's metadata.
    #[must_use]
    pub fn matches(&self, md: &Metadata) -> bool {
        match self {
            Filter::Any => true,
            Filter::Eq(k, v) => md.get(k).is_some_and(|x| x == v),
            Filter::Ne(k, v) => md.get(k) != Some(v),
            Filter::Lt(k, b) => numeric(md, k).is_some_and(|x| x < *b),
            Filter::Le(k, b) => numeric(md, k).is_some_and(|x| x <= *b),
            Filter::Gt(k, b) => numeric(md, k).is_some_and(|x| x > *b),
            Filter::Ge(k, b) => numeric(md, k).is_some_and(|x| x >= *b),
            Filter::In(k, vs) => md.get(k).is_some_and(|x| vs.contains(x)),
            Filter::Exists(k) => md.get(k).is_some_and(|x| x != &MetaValue::Null),
            Filter::Missing(k) => md.get(k).map_or(true, |x| x == &MetaValue::Null),
            Filter::And(fs) => fs.iter().all(|f| f.matches(md)),
            Filter::Or(fs) => fs.iter().any(|f| f.matches(md)),
            Filter::Not(f) => !f.matches(md),
        }
    }
}

/// Reads field `k` of `md` as an `f64`, if it is present and numeric.
fn numeric(md: &Metadata, k: &str) -> Option<f64> {
    md.get(k).and_then(MetaValue::as_f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Metadata {
        Metadata::from([
            ("lang".into(), MetaValue::from("rust")),
            ("year".into(), MetaValue::from(2026_i64)),
            ("score".into(), MetaValue::from(0.9_f64)),
            ("active".into(), MetaValue::from(true)),
        ])
    }

    #[test]
    fn equality_and_membership() {
        let md = sample();
        assert!(Filter::Eq("lang".into(), "rust".into()).matches(&md));
        assert!(!Filter::Eq("lang".into(), "python".into()).matches(&md));
        assert!(Filter::In("lang".into(), vec!["go".into(), "rust".into()]).matches(&md));
    }

    #[test]
    fn numeric_comparisons_coerce_ints_and_bools() {
        let md = sample();
        assert!(Filter::Ge("year".into(), 2020.0).matches(&md));
        assert!(Filter::Lt("year".into(), 2030.0).matches(&md));
        assert!(Filter::Gt("score".into(), 0.5).matches(&md));
        // Strings are not numeric → comparison never matches.
        assert!(!Filter::Gt("lang".into(), 0.0).matches(&md));
    }

    #[test]
    fn presence_and_negation() {
        let md = sample();
        assert!(Filter::Exists("lang".into()).matches(&md));
        assert!(Filter::Missing("author".into()).matches(&md));
        assert!(!Filter::Eq("lang".into(), "rust".into())
            .negate()
            .matches(&md));
    }

    #[test]
    fn boolean_composition() {
        let md = sample();
        let f = Filter::all([
            Filter::Eq("active".into(), true.into()),
            Filter::any([
                Filter::Eq("lang".into(), "go".into()),
                Filter::Ge("year".into(), 2025.0),
            ]),
        ]);
        assert!(f.matches(&md));
    }
}
