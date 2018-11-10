use std::mem::size_of;

pub fn sum(v: &[f64]) -> f64 {
    let mut o = 0.;
    for e in v.iter() {
        o += *e;
    }
    o
}

pub fn mean(v: &[f64]) -> f64 {
    sum(&v) / (v.len() as f64)
}

pub fn prod_vec(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(&a, &b)| a * b).collect()
}

pub fn rmse(target: &[f64], yhat: &[f64]) -> f64 {
    let rmse: f64 = yhat
        .iter()
        .zip(target.iter())
        .map(|(&a, &b)| (a - b).powi(2))
        .sum();
    (rmse / target.len() as f64).sqrt()
}

pub(crate) fn sum_indices(v: &[f64], indices: &[usize]) -> f64 {
    // A sum over a null set is not possible there, and this catch bugs.
    // The speed difference is negligible
    assert_ne!(indices.len(), 0);
    let mut o = 0.;
    for &i in indices {
        o += v[i];
    }
    o
}

pub(crate) fn transmute_vec<T: Sized + Clone>(v: &mut [u8]) -> &mut [T] {
    assert_eq!(v.len() % size_of::<T>(), 0);
    unsafe { std::slice::from_raw_parts_mut(v.as_ptr() as *mut T, v.len() / size_of::<T>()) }
}

#[allow(dead_code)]
pub(crate) fn split_at_mut_transmute<T: Sized + Clone>(
    v: &mut [u8],
    n_elements: usize,
) -> (&mut [T], &mut [u8]) {
    let n_bytes = n_elements * size_of::<T>();
    assert!(
        v.len() >= n_bytes,
        "cache too small, got {}, expected {}",
        v.len(),
        n_bytes
    );
    let (a, b) = v.split_at_mut(n_bytes);
    (transmute_vec(a), b)
}

#[allow(dead_code)]
pub(crate) fn sub_vec(a: &mut [f64], b: &[f64]) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter_mut().zip(b.iter()) {
        *a -= b;
    }
}

pub(crate) fn cosine_simularity(a: &[f64], b: &[f64]) -> f64 {
    let (mut ab, mut a2, mut b2) = (0., 0., 0.);
    for (&a, &b) in a.iter().zip(b) {
        ab += a * b;
        a2 += a.powi(2);
        b2 += b.powi(2);
    }
    ab / (a2 * b2).sqrt()
}

/// Minimize a - x*b
pub(crate) fn min_diff_vectors(a: &[f64], b: &[f64]) -> f64 {
    // We want to minimize d(x) = sum_i (target_i - x * prediction_i)**2
    // Min is at x* = (d(-1)-d(1)) / 2 / (d(1) + d(-1) - 2*d(0))
    let (mut d0, mut d1, mut d1_) = (0., 0., 0.);
    for (target, prediction) in a.iter().zip(b.iter()) {
        d0 += target.powi(2);
        d1 += (target - prediction).powi(2);
        d1_ += (target + prediction).powi(2);
    }
    (d1_ - d1) / (2. * (d1 + d1_ - 2. * d0)) / 2.
}

#[cfg(test)]
mod tests {
    use crate::*;

    macro_rules! assert_almost_eq {
        ($a : expr, $b:expr) => {
            let (a, b) = ($a, $b);
            let eps = 1e-5;
            let diff = (a - b).abs();
            if diff > eps {
                panic!("{} != {} at +-{}", a, b, eps)
            }
        };
    }

    #[test]
    fn test_cosine_similarity() {
        assert_almost_eq!(cosine_simularity(&vec![1., 2., 3.], &vec![1., 2., 3.]), 1.);
        assert_almost_eq!(
            cosine_simularity(&vec![1., 2., 3.], &vec![4., 7., 0.]),
            0.59669419
        );
    }
}
