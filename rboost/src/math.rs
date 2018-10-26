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
