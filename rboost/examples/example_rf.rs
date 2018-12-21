#![feature(duration_as_u128)]

// Example of a regression with a random forest.

extern crate cpuprofiler;
extern crate csv;
extern crate rboost;
extern crate serde_json;

use cpuprofiler::PROFILER;
use rand::prelude::{SeedableRng, SmallRng};
use rboost::{parse_csv, rmse, RFParams, RandomForest, RegLoss, TreeParams};
use std::fs::File;
use std::io::Write;
use std::time::Instant;

fn main() -> Result<(), Box<::std::error::Error>> {
    // Load the data
    let train = include_str!("../data/regression.train");
    let train = parse_csv(train, "\t")?;
    let test = include_str!("../data/regression.test");
    let test = parse_csv(test, "\t")?;

    // Random forest is a stochastic algorithm. For better control you can set the seed before
    // or use `let mut rng = ::rand::thread_rng()`
    let seed = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]; // byte array
    let mut rng = SmallRng::from_seed(seed);

    // We set the params for the RF and the trees
    let mut rf_params = RFParams::new();
    rf_params.colsample_bytree = 1.0;
    rf_params.n_trees = 100;
    let tree_params = TreeParams {
        gamma: 1.,
        lambda: 1.,
        max_depth: 10,
        min_split_gain: 1.,
    };

    // We use binning so the training is much faster.
    let n_bins = 256;

    println!(
        "Params rf={:?} tree{:?} n_bins={}",
        rf_params, tree_params, n_bins
    );

    // cpuprofiler allows us to profile the hot loop
    let profile_path = "./example_rf.profile";
    println!("Profiling to {}", profile_path);
    PROFILER.lock()?.start(profile_path)?;
    let predict_start_time = Instant::now();

    let (rf, yhat_cv) = RandomForest::build(
        // Important: we have to transform the dataset to a PreparedDataset.
        // This step could be done just once if you want to train multiple RF.
        &mut train.as_prepared_data(n_bins)?,
        &rf_params,
        &tree_params,
        RegLoss::default(),
        &mut rng,
    )?;

    println!(
        "{} RF fit. Elapsed: {:.2} secs",
        rf_params.n_trees,
        predict_start_time.elapsed().as_nanos() as f64 / 1_000_000_000.
    );
    PROFILER.lock()?.stop()?;

    // RF gives us direct cross-validated predictions. It's usually a little bit worse than test
    // because we use only 1-1/e = 63% of the train set.
    println!("RMSE train CV {:.8}", rmse(&train.target, &yhat_cv));

    let predict_start_time = Instant::now();
    let yhat_train: Vec<f64> = (0..train.features.n_rows())
        .map(|i| rf.predict(&train.features.row(i)))
        .collect();
    println!(
        "Predictions done in {:.2} secs",
        predict_start_time.elapsed().as_nanos() as f64 / 1_000_000_000.
    );
    println!("RMSE train {:.8}", rmse(&train.target, &yhat_train));

    let yhat_test: Vec<f64> = (0..test.features.n_rows())
        .map(|i| rf.predict(&test.features.row(i)))
        .collect();
    println!("RMSE Test {:.8}", rmse(&test.target, &yhat_test));

    println!("Serializing model to example_rf.json");
    let serialized: String = serde_json::to_string(&rf)?;
    let mut file = File::create("example_rf.json")?;
    file.write_all(serialized.as_bytes())?;

    println!("Writing predictions to example_rf.csv");
    let file = File::create("example_rf.csv")?;
    let mut wtr = csv::Writer::from_writer(file);
    wtr.write_record(&["dataset", "true_val", "yhat"])?;
    for (true_val, yhat) in train.target.iter().zip(yhat_train.iter()) {
        wtr.write_record(&["train", &true_val.to_string(), &yhat.to_string()])?;
    }
    for (true_val, yhat) in test.target.iter().zip(yhat_test.iter()) {
        wtr.write_record(&["test", &true_val.to_string(), &yhat.to_string()])?;
    }
    wtr.flush()?;
    Ok(())
}
