use crate::sum;

/// General interface for a loss.
pub trait Loss: std::marker::Sync {
    fn calc_gradient_hessian(&self, target: &[f64], predictions: &[f64]) -> (Vec<f64>, Vec<f64>);
    fn calc_loss(&self, target: &[f64], predictions: &[f64]) -> f64;

    /// Transform from latent variables (eg. odd for logistic regression) to the target (eg. proba)
    fn get_target(&self, latent: f64) -> f64;

    /// Initial value for the prediction. Useful for limiting bias in regression and unbalanced
    /// classes for classification.
    fn get_initial_prediction(&self, target: &[f64]) -> f64;
}

/// L2 Loss, ie the usual loss for a regression.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RegLoss {
    // Nothing inside
}

impl Default for RegLoss {
    fn default() -> Self {
        RegLoss {}
    }
}

impl Loss for RegLoss {
    fn calc_gradient_hessian(&self, target: &[f64], predictions: &[f64]) -> (Vec<f64>, Vec<f64>) {
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
        sum(&errors)
    }

    fn get_target(&self, latent: f64) -> f64 {
        latent
    }

    fn get_initial_prediction(&self, target: &[f64]) -> f64 {
        target.iter().sum::<f64>() / (target.len() as f64)
    }
}

/// Binary log loss, for two-class classification
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct BinaryLogLoss {
    // Nothing inside
}

impl Default for BinaryLogLoss {
    fn default() -> Self {
        BinaryLogLoss {}
    }
}

impl Loss for BinaryLogLoss {
    fn calc_gradient_hessian(&self, target: &[f64], predictions: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let mut hessian: Vec<f64> = Vec::with_capacity(target.len());
        let mut grad = Vec::with_capacity(target.len());
        for (&target, &latent) in target.iter().zip(predictions.iter()) {
            let proba = self.get_target(-latent);
            grad.push(proba - target);
            hessian.push(proba * (1. - proba));
        }
        (grad, hessian)
    }

    // Targets must be 0 or 1 exactly - for the moment. This loss function could generalize to
    // y in [0,1]
    #[allow(clippy::float_cmp)]
    fn calc_loss(&self, target: &[f64], predictions: &[f64]) -> f64 {
        // target y = 0 or 1
        // proba p = 1 / (1 + exp(-x))
        // -Loss = - y * log(p) - (1-y) * log(1-p)
        //       = y * log(1+exp(-x)) - (1-y) * log(exp(-x)) + (1-y) *  log(1+exp(-x))
        //       = log(1+exp(-x)) + (1-y) * x
        // See https://play.rust-lang.org/?edition=2015&gist=1b59228c7a9ceea83a769b644927f192
        // for benchmarks (including Taylor series)
        let mut errors = Vec::new();
        for (&y, &x) in target.iter().zip(predictions.iter()) {
            assert!((y == 0.) | (y == 1.), "Target must be 0 or 1, got {}", y);
            let loss = (1. - y) * x + (-x).exp().ln_1p();
            errors.push(loss);
        }
        sum(&errors)
    }

    fn get_target(&self, latent: f64) -> f64 {
        1. / (1. + latent.exp())
    }

    fn get_initial_prediction(&self, target: &[f64]) -> f64 {
        let mean_val = (target.iter().sum::<f64>()) / (target.len() as f64);
        let mean_val = mean_val.max(1e-8).min(1. - 1e-8);
        (mean_val / (1. - mean_val)).ln()
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

    macro_rules! assert_close {
        ($a : expr, $b: expr, $delta: expr) => {{
            let (a, b, delta) = ($a, $b, $delta);
            assert!(
                (a - b).abs() <= delta,
                "Difference = {:.6} > {:.6} too important between {:.6} and {:.6}",
                a - b,
                delta,
                a,
                b
            );
        }};
    }

    #[test]
    fn test_reg_loss() {
        let target = vec![0.1, 0.4, 1.3, 0.2];
        let predictions = vec![0.1, 0.4, 1.0, 0.0];
        let eps = 1e-5;
        let loss_reg = RegLoss::default();

        let loss = loss_reg.calc_loss(&target, &predictions);

        let (grad, hessian) = loss_reg.calc_gradient_hessian(&target, &predictions);
        for i in 0..target.len() {
            // Test gradient
            // f'(x) = (f(x+eps) - f(x-eps)) / (2*eps)
            let l_plus = loss_reg.calc_loss(&inc_vec(&target, i, eps), &predictions);
            let l_minus = loss_reg.calc_loss(&inc_vec(&target, i, -eps), &predictions);
            let grad_emp = (l_plus - l_minus) / (2. * eps);
            assert_close!(grad[i], grad_emp, 1e-5);

            // Test hessian
            // f"(x) = (f'(x+eps/2) - f'(x-eps/2)) / (2*eps/2)
            //       = (f(x+eps)-f(x) - f(x) + f(x-eps)) / (eps*eps)
            let hessian_emp = (l_plus + l_minus - 2. * loss) / eps.powi(2);
            assert_close!(hessian[i], hessian_emp, 1e-5);
        }
    }

    //noinspection RsApproxConstant
    #[test]
    fn test_binary_loss() {
        let target = vec![1., 1., 0., 1.];
        let predictions = vec![0.1, 0.4, 0.9, 0.3];
        let eps = 1e-5;
        let loss_reg = BinaryLogLoss::default();

        let loss = loss_reg.calc_loss(&target, &predictions);
        let expected = 2.9529210316741383;
        assert_close!(loss, expected, 1e-3);

        let (grad, hessian) = loss_reg.calc_gradient_hessian(&target, &predictions);
        for i in 0..target.len() {
            // Test gradient
            // f'(x) = (f(x+eps) - f(x-eps)) / (2*eps)
            let l_plus = loss_reg.calc_loss(&target, &inc_vec(&predictions, i, eps));
            let l_minus = loss_reg.calc_loss(&target, &inc_vec(&predictions, i, -eps));
            let grad_emp = (l_plus - l_minus) / (2. * eps);
            assert_close!(grad[i], grad_emp, 1e-5);

            // Test hessian
            // f"(x) = (f'(x+eps/2) - f'(x-eps/2)) / (2*eps/2)
            //       = (f(x+eps)-f(x) - f(x) + f(x-eps)) / (eps*eps)
            let hessian_emp = (l_plus + l_minus - 2. * loss) / eps.powi(2);
            assert_close!(hessian[i], hessian_emp, 1e-5);
        }
    }
}
