pub trait Loss {
    fn calc_gradient(&self, target: &[f64], predictions: &[f64]) -> (Vec<f64>, Vec<f64>);
    fn calc_loss(&self, target: &[f64], predictions: &[f64]) -> f64;
}

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

pub struct RegLoss {
    // Nothing inside
}

impl Default for RegLoss {
    fn default() -> Self {
        RegLoss {}
    }
}

impl Loss for RegLoss {
    fn calc_gradient(&self, target: &[f64], predictions: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let hessian: Vec<f64> = (0..target.len()).map(|_| 2.).collect();
        let grad = (0..target.len())
            .map(|i| 2. * (target[i] - predictions[i]))
            .collect();
        (grad, hessian)
    }

    fn calc_loss(&self, target: &[f64], predictions: &[f64]) -> f64 {
        let mut errors = Vec::new();
        for (n_row, &target) in target.iter().enumerate() {
            let diff = target - predictions[n_row];
            errors.push(diff.powi(2));
        }
        return sum(&errors);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    /// Clone the vector and increase a given index
    fn inc_vec(v: &Vec<f64>, i: usize, eps: f64) -> Vec<f64> {
        let mut v = v.clone();
        v[i] += eps;
        v
    }

    fn assert_close(a: f64, b: f64, delta: f64) {
        assert!(
            (a - b).abs() <= delta,
            "Difference = {:.6} > {:.6} too important between {:.6} and {:.6}",
            a - b,
            delta,
            a,
            b
        );
    }

    #[test]
    fn test_reg_loss() {
        let target = vec![0.1, 0.4, 1.3, 0.2];
        let predictions = vec![0.1, 0.4, 1.0, 0.0];
        let eps = 1e-5;
        let loss_reg = RegLoss::default();

        let loss = loss_reg.calc_loss(&target, &predictions);

        let (grad, hessian) = loss_reg.calc_gradient(&target, &predictions);
        for i in 0..target.len() {
            // Test gradient
            // f'(x) = (f(x+eps) - f(x-eps)) / (2*eps)
            let l_plus = loss_reg.calc_loss(&inc_vec(&target, i, eps), &predictions);
            let l_minus = loss_reg.calc_loss(&inc_vec(&target, i, -eps), &predictions);
            let grad_emp = (l_plus - l_minus) / (2. * eps);
            assert_close(grad[i], grad_emp, 1e-5);

            // Test hessian
            // f"(x) = (f'(x+eps/2) - f'(x-eps/2)) / (2*eps/2)
            //       = (f(x+eps)-f(x) - f(x) + f(x-eps)) / (eps*eps)
            let hessian_emp = (l_plus + l_minus - 2. * loss) / eps.powi(2);
            assert_close(hessian[i], hessian_emp, 1e-5);
        }
    }
}
