#![feature(duration_as_u128)]

/// WARNING: boosting is NOT ready, do NOT use it for real work
extern crate cpuprofiler;
extern crate csv;
extern crate failure;
extern crate rboost;
extern crate serde_json;

use cpuprofiler::PROFILER;
use rand::prelude::{SeedableRng, SmallRng};
use rboost::{parse_csv, BinaryLogLoss, Booster, BoosterParams, TreeParams, GBT};
use rustlearn::prelude::Array;

fn roc_auc_score(y_true: &[f64], y_hat: &[f64]) -> Result<f32, Box<::std::error::Error>> {
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
    Ok(::rustlearn::metrics::roc_auc_score(&y_true, &y_hat)?)
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

fn main() -> Result<(), Box<::std::error::Error>> {
    let train = include_str!("../data/binary.train");
    let train = parse_csv(train, "\t")?;
    let test = include_str!("../data/binary.test");
    let test = parse_csv(test, "\t")?;

    let seed = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]; // byte array
    let mut rng = SmallRng::from_seed(seed);

    let booster_params = BoosterParams {
        learning_rate: 1.,
        booster: Booster::Geometric,
        colsample_bytree: 1.,
    };
    let tree_params = TreeParams {
        gamma: 0.,
        lambda: 1.,
        max_depth: 8,
        min_split_gain: 0.,
    };
    let n_bins = 2048;
    println!(
        "Params booster={:?} tree{:?} n_bins={}",
        booster_params, tree_params, n_bins
    );
    println!("Profiling to example_boost_binary.profile");
    PROFILER.lock()?.start("./example_boost_binary.profile")?;
    let gbt = GBT::build(
        &booster_params,
        &tree_params,
        &mut train.as_prepared_data(n_bins)?,
        100,
        Some(&test),
        10000,
        BinaryLogLoss::default(),
        &mut rng,
    )?;
    PROFILER.lock()?.stop()?;

    let yhat_train: Vec<f64> = (0..train.features.n_rows())
        .map(|i| gbt.predict(&train.features.row(i)))
        .collect();
    println!(
        "TRAIN: ROC AUC {:.8}, accuracy {:.8}",
        roc_auc_score(&train.target, &yhat_train)?,
        accuracy_score(&train.target, &yhat_train),
    );

    let yhat_test: Vec<f64> = (0..test.features.n_rows())
        .map(|i| gbt.predict(&test.features.row(i)))
        .collect();
    println!(
        "TEST:  ROC AUC {:.8}, accuracy {:.8}",
        roc_auc_score(&test.target, &yhat_test)?,
        accuracy_score(&test.target, &yhat_test),
    );
    Ok(())
}
