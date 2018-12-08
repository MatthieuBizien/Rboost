#![feature(duration_as_u128)]
#![feature(plugin, custom_attribute)]
#![feature(inner_deref)]

extern crate core;
extern crate failure;
extern crate ord_subset;
extern crate ordered_float;
extern crate rand;
extern crate rayon;
#[macro_use]
extern crate serde_derive;

mod dart;
mod data;
#[allow(dead_code)]
mod gbt;
mod losses;
mod math;
mod matrix;
mod rf;
mod tree;
mod tree_bin;
mod tree_direct;

pub use crate::dart::*;
pub use crate::data::*;
#[doc(hidden)] // TODO implements boosting correctly
pub use crate::gbt::*;
pub use crate::losses::*;
pub use crate::math::*;
pub use crate::matrix::*;
pub use crate::rf::*;
pub use crate::tree::*;

pub(crate) static DEFAULT_GAMMA: f64 = 0.;
pub(crate) static DEFAULT_LAMBDA: f64 = 1.;
pub(crate) static DEFAULT_LEARNING_RATE: f64 = 0.8;
pub(crate) static DEFAULT_MAX_DEPTH: usize = 3;
pub(crate) static DEFAULT_MIN_SPLIT_GAIN: f64 = 0.1;
pub(crate) static DEFAULT_COLSAMPLE_BYTREE: f64 = 1.;
pub(crate) static DEFAULT_N_TREES: usize = 1000;
