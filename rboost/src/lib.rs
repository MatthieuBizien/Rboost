#[macro_use]
extern crate serde_derive;

mod dart;
mod data;
mod error;
mod gbt;
mod losses;
mod math;
mod matrix;
mod rf;
mod tree;
mod tree_all;
mod tree_bin;
mod tree_direct;

pub use crate::dart::*;
pub use crate::data::*;
pub use crate::error::*;
#[doc(hidden)] // TODO implements boosting correctly
pub use crate::gbt::*;
pub use crate::losses::*;
pub use crate::math::*;
pub use crate::matrix::dense::*;
pub use crate::rf::*;
pub use crate::tree::*;

pub(crate) static DEFAULT_GAMMA: f64 = 0.;
pub(crate) static DEFAULT_LAMBDA: f64 = 1.;
pub(crate) static DEFAULT_LEARNING_RATE: f64 = 0.1;
pub(crate) static DEFAULT_MAX_DEPTH: usize = 3;
pub(crate) static DEFAULT_MIN_SPLIT_GAIN: f64 = 0.1;
pub(crate) static DEFAULT_COLSAMPLE_BYTREE: f64 = 1.;
pub(crate) static DEFAULT_N_TREES: usize = 1000;

#[doc(hidden)]
/// Transform a duration to the number of sec in float.
/// Useful for the examples while duration_float is not stable.
pub fn duration_as_f64(duration: &::std::time::Duration) -> f64 {
    let nano = duration.subsec_nanos() as f64;
    let sec = duration.as_secs() as f64;
    sec + nano / 1_000_000_000.
}
