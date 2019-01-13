extern crate cpuprofiler;
extern crate csv;
extern crate rboost;
extern crate serde_json;

use cpuprofiler::PROFILER;
use rand::prelude::{SeedableRng, SmallRng};
use rboost::duration_as_f64;
use rboost::{
    accuracy_score, parse_csv, roc_auc_score, BinaryLogLoss, Dart, DartParams, TreeParams,
};
use std::time::Instant;

fn main() -> Result<(), Box<::std::error::Error>> {
    let train = include_str!("../data/binary.train");
    let train = parse_csv(train, "\t")?;
    let test = include_str!("../data/binary.test");
    let test = parse_csv(test, "\t")?;

    let seed = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]; // byte array
    let mut rng = SmallRng::from_seed(seed);

    let mut booster_params = DartParams::default();
    booster_params.colsample_bytree = 0.95;
    booster_params.dropout_rate = 0.5;
    booster_params.learning_rate = 0.3;

    let mut tree_params = TreeParams::default();
    tree_params.max_depth = 5;
    tree_params.min_split_gain = 10.;

    let n_bins = 256;
    println!(
        "Params booster={:?} tree{:?} n_bins={}",
        booster_params, tree_params, n_bins
    );

    println!("Profiling to example_dart_binary.profile");
    PROFILER.lock()?.start("./example_dart_binary.profile")?;
    let train_start_time = Instant::now();
    let gbt = Dart::build(
        &booster_params,
        &tree_params,
        &mut train.as_prepared_data(n_bins)?,
        100,
        Some(&test),
        100,
        BinaryLogLoss::default(),
        &mut rng,
    )?;
    println!(
        "Training of {} trees finished. Elapsed: {:.2} secs",
        gbt.n_trees(),
        duration_as_f64(&train_start_time.elapsed()),
    );
    PROFILER.lock()?.stop()?;

    let yhat_train: Vec<f64> = (0..train.features.n_rows())
        .map(|i| gbt.predict(&train.features.row(i)))
        .collect();
    println!(
        "TRAIN: ROC AUC {:.8}, accuracy {:.8}",
        roc_auc_score(&train.target, &yhat_train)?,
        accuracy_score(&train.target, &yhat_train)?,
    );

    let yhat_test: Vec<f64> = (0..test.features.n_rows())
        .map(|i| gbt.predict(&test.features.row(i)))
        .collect();
    println!(
        "TEST:  ROC AUC {:.8}, accuracy {:.8}",
        roc_auc_score(&test.target, &yhat_test)?,
        accuracy_score(&test.target, &yhat_test)?,
    );
    Ok(())
}
