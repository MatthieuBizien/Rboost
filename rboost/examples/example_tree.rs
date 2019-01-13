extern crate cpuprofiler;
extern crate csv;
extern crate rboost;
extern crate serde_json;

use rboost::{parse_csv, rmse, BinaryLogLoss, DecisionTree, TreeParams};

fn main() -> Result<(), Box<::std::error::Error>> {
    // Load the data
    let train = include_str!("../data/regression.train");
    let train = parse_csv(train, "\t")?;
    let test = include_str!("../data/regression.test");
    let test = parse_csv(test, "\t")?;

    let n_bins = 128;

    for max_depth in 0..8 {
        let mut tree_params = TreeParams::default();
        tree_params.max_depth = max_depth;
        tree_params.min_split_gain = 0.;

        println!("\nParams tree{:?} n_bins={}", tree_params, n_bins);

        let mut predictions = vec![0.; train.target.len()];
        let tree = DecisionTree::build(
            &mut train.as_prepared_data(n_bins)?,
            &mut predictions,
            &tree_params,
            BinaryLogLoss::default(),
        )?;

        let yhat_train: Vec<f64> = (0..train.features.n_rows())
            .map(|i| tree.predict(&train.features.row(i)))
            .collect();
        println!("RMSE train {:.8}", rmse(&train.target, &yhat_train));

        let yhat_test: Vec<f64> = (0..test.features.n_rows())
            .map(|i| tree.predict(&test.features.row(i)))
            .collect();
        println!("RMSE Test {:.8}", rmse(&test.target, &yhat_test));
    }

    Ok(())
}
