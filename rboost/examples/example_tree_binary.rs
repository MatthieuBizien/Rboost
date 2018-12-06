#![feature(duration_as_u128)]

/// WARNING: boosting is NOT ready, do NOT use it for real work
extern crate cpuprofiler;
extern crate csv;
extern crate failure;
extern crate rboost;
extern crate serde_json;

use rboost::{parse_csv, BinaryLogLoss, DecisionTree, TreeParams};
use rustlearn::prelude::Array;

fn roc_auc_score(y_true: &[f64], y_hat: &[f64]) -> f32 {
    for &y in y_true {
        assert!((y == 0.) | (y == 1.), "Target must be 0 or 1, got {}", y)
    }
    for &proba in y_hat {
        assert!(
            (proba >= 0.) & (proba <= 1.),
            "Prediction must be between 0 and 1, got {}",
            proba
        )
    }
    let y_true: Vec<_> = y_true.iter().map(|e| *e as f32).collect();
    let y_true: Array = Array::from(y_true);
    let y_hat: Vec<_> = y_hat.iter().map(|e| *e as f32).collect();
    let y_hat: Array = Array::from(y_hat);
    ::rustlearn::metrics::roc_auc_score(&y_true, &y_hat).expect("Error on ROC AUC")
}

fn accuracy_score(y_true: &[f64], y_hat: &[f64]) -> f32 {
    let mut n_ok = 0;
    for (&a, &b) in y_true.iter().zip(y_hat) {
        let a = if a == 0. {
            false
        } else if a == 1. {
            true
        } else {
            panic!("Label must be 0 or 1, got {}", a)
        };
        if a == (b > 0.5) {
            n_ok += 1;
        }
    }
    (n_ok as f32) / (y_true.len() as f32)
}

fn main() {
    let train = include_str!("../data/binary.train");
    let train = parse_csv(train, "\t").expect("Train data");
    let test = include_str!("../data/binary.test");
    let test = parse_csv(test, "\t").expect("Train data");

    let n_bins = 2048;

    for max_depth in 2..10 {
        let tree_params = TreeParams {
            gamma: 0.,
            lambda: 1.,
            max_depth,
            min_split_gain: 0.,
        };
        println!("\nParams tree{:?} n_bins={}", tree_params, n_bins);

        let mut predictions = vec![0.; train.target.len()];
        let tree = DecisionTree::build(
            &mut train.as_prepared_data(n_bins),
            &mut predictions,
            &tree_params,
            BinaryLogLoss::default(),
        );

        let yhat_train: Vec<f64> = (0..train.features.n_rows())
            .map(|i| tree.predict(&train.features.row(i)))
            .collect();
        println!(
            "TRAIN: ROC AUC {:.8}, accuracy {:.8}",
            roc_auc_score(&train.target, &yhat_train),
            accuracy_score(&train.target, &yhat_train),
        );

        let yhat_test: Vec<f64> = (0..test.features.n_rows())
            .map(|i| tree.predict(&test.features.row(i)))
            .collect();
        println!(
            "TEST:  ROC AUC {:.8}, accuracy {:.8}",
            roc_auc_score(&test.target, &yhat_test),
            accuracy_score(&test.target, &yhat_test),
        );
    }
}
