// Source adopted from
// https://github.com/tildeio/helix-website/blob/master/crates/word_count/src/lib.rs
#![feature(specialization)]

#[macro_use]
extern crate pyo3;

use ndarray::Axis;
use ndarray::{ArrayD, ArrayView1, ArrayView2, ArrayViewD, ArrayViewMutD};
use numpy::{IntoPyResult, PyArray1, PyArray2, PyArrayDyn, ToPyArray};
use pyo3::prelude::*;
use rboost::Params;
use std::fs;
use std::path::PathBuf;

/// Represents a file that can be searched
#[pyclass]
struct RBoostRegressor {
    params: Params,
}

#[pymethods]
impl RBoostRegressor {
    #[new]
    fn __new__(
        obj: &PyRawObject,
        gamma: Option<f64>,
        lambda: Option<f64>,
        learning_rate: Option<f64>,
        max_depth: Option<usize>,
        min_split_gain: Option<f64>,
    ) -> PyResult<()> {
        let mut params = Params::new();
        if let Some(gamma) = gamma {
            params.gamma = gamma;
        }
        if let Some(lambda) = lambda {
            params.lambda = lambda;
        }
        if let Some(learning_rate) = learning_rate {
            params.learning_rate = learning_rate;
        }
        if let Some(max_depth) = max_depth {
            params.max_depth = max_depth;
        }
        if let Some(min_split_gain) = min_split_gain {
            params.min_split_gain = min_split_gain;
        }
        obj.init(|_| RBoostRegressor { params: params })
    }

    /// Searches for the word, parallelized by rayon
    fn fit(&self, py: Python, x: &PyArray2<f64>, y: &PyArray1<f64>) -> PyResult<f64> {
        let x = x.as_array()?;
        let y = y.as_array()?;
        let target: Vec<f64> = y.iter().map(|e| (*e).clone()).collect();
        let features: Vec<Vec<f64>> = (0..x.rows())
            .map(|col| {
                x.subview(Axis(1), col)
                    .iter()
                    .map(|e| (*e).clone())
                    .collect()
            })
            .collect();
        Ok(y.scalar_sum() + x.scalar_sum())
    }
}

#[pymodinit]
fn rboost_python(_py: Python, m: &PyModule) -> PyResult<()> {
    #[pyfn(m, "fit")]
    fn fit_py(x: &PyArray2<f64>, y: &PyArray1<f64>) -> PyResult<f64> {
        let x = x.as_array()?;
        let y = y.as_array()?;
        let target: Vec<f64> = y.iter().map(|e| (*e).clone()).collect();
        let features: Vec<Vec<f64>> = (0..x.rows())
            .map(|col| {
                x.subview(Axis(1), col)
                    .iter()
                    .map(|e| (*e).clone())
                    .collect()
            })
            .collect();
        Ok(y.scalar_sum() + x.scalar_sum())
    }

    m.add_class::<RBoostRegressor>()?;

    Ok(())
}
