use data::Matrix;
use math::{cumulative_sum, rolling_var};

type Float = f32;
type Target = f64;

pub struct ResultRow {
    pub row: usize,
    pub error: Float,
    pub grad: Float,
    pub hessian: Float,
}

pub struct Result {
    pub error: Float,
    pub col: usize,
    pub left_results: Vec<ResultRow>,
    pub right_results: Vec<ResultRow>,
}

pub trait Objective {
    fn error_col(&self, col: usize, matrix: &Matrix) -> Result;
}

pub struct LinearModel;

impl Objective for LinearModel {
    #[allow(unused)]
    fn error_col(&self, col: usize, matrix: &Matrix) -> Result {
        let (rows, vals) = matrix.sorted_col(col);
        let targets: Vec<_> = rows.iter().map(|&row| matrix.get(row, col)).collect();
        let mut global_error = 0.;

        // Rule: we want the split where the variance of the splits are the lower
        let labels = rows.iter().map(|&row| matrix.get_label(row)).collect();

        let variances = rolling_var(&labels);
        let labels_rev = labels.clone().into_iter().rev().collect();
        let mut variances_rev = (rolling_var(&labels_rev));
        variances_rev.reverse();

        println!("labels {:?}", labels);
        println!("labels_rev {:?}", labels_rev);
        println!("variances {:?}", variances);
        println!("variances_rev {:?}", variances_rev);

        let get_var = |n: usize| {
            variances[n].1 + variances_rev[n].1 + (variances[n].0 - variances[n].0).powi(2)
        };

        let mut best_split = 0;
        let mut current_best = get_var(0);
        for n in 1..variances.len() {
            let var: Target = get_var(n);
            println!("var {:.2} \t best={:.2}", var, current_best);
            if var < current_best {
                best_split = n;
                current_best = var;
            }
        }
        println!("best_split {} current_best {}", best_split, current_best);
        //let global_var: Vec<_> = variances.iter().zip(variances_rev).map(|(&a, b)|a+b).collect();
        unimplemented!();
        /*
        let mut left_results = Vec::new();
        let mut right_results = Vec::new();
        for (ix, (val, target)) in vals.into_iter().zip(targets).enumerate() {
            let grad = target - mean;
            let error = grad.powi(2);
            let row = rows[ix];
            let result = ResultRow {
                row: matrix.rows[ix],
                error,
                grad,
                hessian: 1.,
            };
            global_error += error;

            if target <= mean {
                left_results.push(result);
            } else {
                right_results.push(result)
            }
        }
        Result {
            error: global_error,
            col,
            left_results,
            right_results,
        }
        */
    }
}
