pub trait Loss {
    fn calc_gradient(&self, target: &[f64], predictions: &[f64]) -> (Vec<f64>, Vec<f64>);
    fn calc_loss(&self, target: &[f64], predictions: &[f64]) -> f64;
}

fn sum(v: &[f64]) -> f64 {
    let mut o = 0.;
    for e in v.iter() {
        o += *e;
    }
    o
}

fn mean(v: &[f64]) -> f64 {
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
        return mean(&errors);
    }
}
