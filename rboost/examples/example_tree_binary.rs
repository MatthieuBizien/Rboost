#![feature(duration_as_u128)]

/// WARNING: boosting is NOT ready, do NOT use it for real work
extern crate cpuprofiler;
extern crate csv;
extern crate rboost;
extern crate serde_json;

use rboost::{accuracy_score, parse_csv, roc_auc_score, BinaryLogLoss, DecisionTree, TreeParams};

fn main() -> Result<(), Box<::std::error::Error>> {
    let train = include_str!("../data/binary.train");
    let train = parse_csv(train, "\t")?;
    let test = include_str!("../data/binary.test");
    let test = parse_csv(test, "\t")?;

    let n_bins = 2048;

    for max_depth in 2..10 {
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
        println!(
            "TRAIN: ROC AUC {:.8}, accuracy {:.8}",
            roc_auc_score(&train.target, &yhat_train)?,
            accuracy_score(&train.target, &yhat_train)?,
        );

        let yhat_test: Vec<f64> = (0..test.features.n_rows())
            .map(|i| tree.predict(&test.features.row(i)))
            .collect();
        println!(
            "TEST:  ROC AUC {:.8}, accuracy {:.8}",
            roc_auc_score(&test.target, &yhat_test)?,
            accuracy_score(&test.target, &yhat_test)?,
        );
    }

    Ok(())
}
