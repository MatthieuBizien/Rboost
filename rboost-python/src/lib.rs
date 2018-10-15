extern crate ndarray;
extern crate numpy;
extern crate pyo3;

use ndarray::Axis;
use ndarray::{ArrayD, ArrayView1, ArrayView2, ArrayViewD, ArrayViewMutD};
use numpy::{IntoPyResult, PyArray1, PyArray2, PyArrayDyn, ToPyArray};
use pyo3::prelude::{pymodinit, PyModule, PyResult, Python};

#[pymodinit]
fn rboost(_py: Python, m: &PyModule) -> PyResult<()> {
    fn fit(x: PyArray2<f64>, y: PyArray1<f64>) -> f64 {
        let x: ArrayView2<f64> = x.as_array().unwrap();
        let y: ArrayView1<f64> = y.as_array().unwrap();
        let target: Vec<f64> = y.iter().map(|e| (*e).clone()).collect();
        let features: Vec<Vec<f64>> = (0..x.rows())
            .map(|col| {
                x.subview(Axis(1), col)
                    .iter()
                    .map(|e| (*e).clone())
                    .collect()
            }).collect();
        y.scalar_sum() + x.scalar_sum()
    }

    // immutable example
    fn axpy(a: f64, x: ArrayViewD<f64>, y: ArrayViewD<f64>) -> ArrayD<f64> {
        a * &x + &y
    }

    // mutable example (no return)
    fn mult(a: f64, mut x: ArrayViewMutD<f64>) {
        x *= a;
    }

    // wrapper of `axpy`
    #[pyfn(m, "axpy")]
    fn axpy_py(
        py: Python,
        a: f64,
        x: &PyArrayDyn<f64>,
        y: &PyArrayDyn<f64>,
    ) -> PyResult<PyArrayDyn<f64>> {
        // you can convert numpy error into PyErr via ?
        let x = x.as_array()?;
        // you can also specify your error context, via closure
        let y = y.as_array().into_pyresult_with(|| "y must be f64 array")?;
        Ok(axpy(a, x, y).to_pyarray(py).to_owned(py))
    }

    // wrapper of `mult`
    #[pyfn(m, "mult")]
    fn mult_py(_py: Python, a: f64, x: &PyArrayDyn<f64>) -> PyResult<()> {
        let x = x.as_array_mut()?;
        mult(a, x);
        Ok(())
    }

    Ok(())
}
