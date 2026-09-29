//! Native ablations of the archived Python experiments.
pub(crate) mod backends;
mod index;
pub mod micro;
pub mod parallel;
mod queue;
pub mod spans;
mod trainer;

pub use trainer::{train_variant, variant_names};

use crate::{Bounds, Prepared, TrainError, TrainOptions, TrainResult};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy)]
pub struct Options {
    pub max_merges: usize,
    pub min_frequency: u64,
    pub bounds: Bounds,
    pub workers: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            max_merges: 3000,
            min_frequency: 2,
            bounds: Bounds::Checked,
            workers: 1,
        }
    }
}

#[derive(Debug)]
pub struct Result {
    pub core: TrainResult,
    pub metrics: BTreeMap<String, f64>,
}

pub(crate) fn validate(input: &Prepared, options: Options) -> std::result::Result<(), TrainError> {
    crate::trainer::validate(
        input,
        TrainOptions {
            max_merges: options.max_merges,
            min_frequency: options.min_frequency,
            bounds: options.bounds,
        },
    )
}
