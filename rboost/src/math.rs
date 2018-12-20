use rand::seq::sample_indices;
use rand::Rng;

pub(crate) fn sum(v: &[f64]) -> f64 {
    let mut o = 0.;
    for e in v.iter() {
        o += *e;
    }
    o
}

pub(crate) fn prod_vec(a: &[f64], b: &[f64]) -> Vec<f64> {
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

#[allow(dead_code)]
pub(crate) fn mul_add(a: &[f64], b: f64, out: &mut [f64]) {
    assert_eq!(a.len(), out.len());
    out.iter_mut().zip(a).for_each(|(out, a)| *out += a * b);
}

#[allow(dead_code)]
pub(crate) fn add(a: &[f64], out: &mut [f64]) {
    assert_eq!(a.len(), out.len());
    out.iter_mut().zip(a).for_each(|(out, a)| *out += a);
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

pub fn roc_auc_score(y_true: &[f64], y_hat: &[f64]) -> Result<f64, Box<::std::error::Error>> {
    if y_hat.len() != y_true.len() {
        Err("Size of inputs are not the same for ROC AUC")?;
    }
    if y_hat.len() == 0 {
        Err("No input for ROC AUC")?;
    }
    let mut v: Vec<_> = y_true
        .iter()
        .zip(y_hat)
        .map(|(&y_true, &y_hat)| (y_true > 0.5, y_hat))
        .collect();
    let auc = classifier_measures::roc_auc_mut(&mut v).ok_or("Error on computation of ROC AUC");
    Ok(auc?)
}

pub fn accuracy_score(y_true: &[f64], y_hat: &[f64]) -> Result<f64, Box<::std::error::Error>> {
    let mut n_ok = 0;
    for (&a, &b) in y_true.iter().zip(y_hat) {
        let a = if a == 0. {
            false
        } else if a == 1. {
            true
        } else {
            Err("Label must be 0 or 1")?;
            unreachable!()
        };
        if a == (b > 0.5) {
            n_ok += 1;
        }
    }
    Ok((n_ok as f64) / (y_true.len() as f64))
}

#[cfg(test)]
mod tests {
    //use crate::*;

    // macro_rules! assert_almost_eq {
    //     ($a : expr, $b:expr) => {
    //         let (a, b) = ($a, $b);
    //         let eps = 1e-5;
    //         let diff = (a - b).abs();
    //         if diff > eps {
    //             panic!("{} != {} at +-{}", a, b, eps)
    //         }
    //     };
    // }
}
