use rand::seq::sample_indices;
use rand::Rng;

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

pub(crate) fn sample_indices_ratio(rng: &mut impl Rng, length: usize, ratio: f64) -> Vec<usize> {
    let n_cols_f64 = ratio * (length as f64);
    let mut n_cols = n_cols_f64.floor() as usize;
    // Randomly select N or N+1 column proportionally to the difference
    if (n_cols_f64 - n_cols as f64) > rng.gen() {
        n_cols += 1;
    }
    sample_indices(rng, length, n_cols)
}

pub(crate) fn weighted_mean(val_1: f64, n_1: usize, val_2: f64, n_2: usize) -> f64 {
    let n = (n_1 + n_2) as f64;
    let (n_1, n_2) = (n_1 as f64, n_2 as f64);
    (val_1 * n_1 + val_2 * n_2) / n
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
