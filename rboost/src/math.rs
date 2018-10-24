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
