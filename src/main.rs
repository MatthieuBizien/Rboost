#![feature(non_ascii_idents)]
#[macro_use]
extern crate assert_approx_eq;

mod data;
mod data_loading;
mod learner;
mod math;
mod objective;

use data::*;
use objective::*;

fn main() {
    let matrix = data_loading::read_csv("boston.csv");
    let linear_objective = objective::LinearModel {};

    for n_col in 1..matrix.num_cols {
        let error = linear_objective.error_col(n_col, &matrix);
        println!("error {}: {}", n_col, error.error);
        break;
    }
}
