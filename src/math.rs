type Float = f32;
type Target = f64;

/// Warning: can be numerically unstable.
pub fn rolling_var(values: &Vec<Target>) -> Vec<(Target, Target)> {
    let mut out = Vec::with_capacity(values.len());

    // We use f64 for numerical stability
    let mut m: f64 = 0.; // sample mean
    let mut σ: f64 = 0.; // population variance

    for (n, &x) in values.iter().enumerate() {
        let x = x as f64;
        let n = (n + 1) as f64;
        let next_m = m + (x - m) / n;
        σ = ((n - 1.) * σ + (x - next_m) * (x - m)) / n;
        m = next_m;
        out.push((m as Target, σ as Target))
    }

    out
}

pub fn cumulative_sum(mut values: Vec<Target>) -> Vec<Target> {
    for n in 1..values.len() {
        values[n] += values[n - 1];
    }
    values
}

macro_rules! assert_2_approx_eq {
    ($a : expr, $b: expr) => {
        let (a0, a1) = $a;
        let (b0, b1) = $b;
        assert_approx_eq!(a0, b0);
        assert_approx_eq!(a1, b1);
    };
}

#[test]
fn using_other_iterator_trait_methods() {
    let vals = vec![1., 2., 4., 8.];
    let l_var = rolling_var(&vals);
    assert_eq!(vals.len(), l_var.len());
    assert_2_approx_eq!(l_var[0], (1., 0.));
    assert_2_approx_eq!(l_var[1], (1.5, 0.25));
    assert_2_approx_eq!(l_var[2], (7. / 3., 1.5555555555555554));
    assert_2_approx_eq!(l_var[3], (15. / 4., 7.1875));
}
