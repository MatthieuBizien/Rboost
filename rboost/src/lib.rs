#![feature(duration_as_u128)]
#![feature(plugin, custom_attribute)]
#![feature(inner_deref)]

extern crate core;
extern crate ord_subset;
extern crate rand;
extern crate rayon;
#[macro_use]
extern crate serde_derive;

mod data;
mod gbt;
mod losses;
mod math;
mod matrix;
mod tree;

pub use crate::data::*;
pub use crate::gbt::*;
pub use crate::losses::*;
pub use crate::math::*;
pub use crate::matrix::*;
use crate::tree::*;

pub use crate::matrix::{ColumnMajorMatrix, StridedVecView};

pub static DEFAULT_GAMMA: f64 = 0.;
pub static DEFAULT_LAMBDA: f64 = 1.;
pub static DEFAULT_LEARNING_RATE: f64 = 0.8;
pub static DEFAULT_MAX_DEPTH: usize = 3;
pub static DEFAULT_MIN_SPLIT_GAIN: f64 = 0.1;
pub static DEFAULT_N_BINS: usize = 256;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Params {
    pub gamma: f64,
    pub lambda: f64,
    pub learning_rate: f64,
    pub max_depth: usize,
    pub min_split_gain: f64,
    pub n_bins: usize,
}

impl Params {
    pub fn new() -> Self {
        Params {
            gamma: DEFAULT_GAMMA,
            lambda: DEFAULT_LAMBDA,
            learning_rate: DEFAULT_LEARNING_RATE,
            max_depth: DEFAULT_MAX_DEPTH,
            min_split_gain: DEFAULT_MIN_SPLIT_GAIN,
            n_bins: DEFAULT_N_BINS,
        }
    }
}
