/// WARNING: boosting is NOT ready, do NOT use it for real work
extern crate cpuprofiler;
extern crate csv;
extern crate rboost;
extern crate serde_json;

use cpuprofiler::PROFILER;
use rand::prelude::{SeedableRng, SmallRng};
use rboost::duration_as_f64;
use rboost::{parse_csv, rmse, BoosterParams, RegLoss, TreeParams, GBT};
use std::fs::File;
use std::io::Write;
use std::time::Instant;

fn main() -> Result<(), Box<::std::error::Error>> {
    let train = include_str!("../data/regression.train");
    let train = parse_csv(train, "\t")?;
    let test = include_str!("../data/regression.test");
    let test = parse_csv(test, "\t")?;

    let seed = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]; // byte array
    let mut rng = SmallRng::from_seed(seed);

    let booster_params = BoosterParams {
        learning_rate: 0.1,
        colsample_bytree: 0.95,
    };

    let mut tree_params = TreeParams::default();
    tree_params.max_depth = 4;
    tree_params.min_split_gain = 0.1;

    let n_bins = 2048;
    println!(
        "Params booster={:?} tree{:?} n_bins={}",
        booster_params, tree_params, n_bins
    );

    println!("Profiling to example_boost.profile");
    PROFILER.lock()?.start("./example_boost.profile")?;
    let train_start_time = Instant::now();
    let gbt = GBT::build(
        &booster_params,
        &tree_params,
        &mut train.as_prepared_data(n_bins)?,
        100,
        Some(&test),
        1000,
        RegLoss::default(),
        &mut rng,
    )?;
    println!(
        "Training of {} trees finished. Elapsed: {:.2} secs",
        gbt.n_trees(),
        duration_as_f64(&train_start_time.elapsed()),
    );
    PROFILER.lock()?.stop()?;

    let n_preds = 10;
    let predict_start_time = Instant::now();
    let mut predictions = vec![0.; train.n_rows()];
    for _ in 0..n_preds {
        for (i, pred) in predictions.iter_mut().enumerate() {
            *pred += gbt.predict(&train.features.row(i));
        }
    }
    println!(
        "{} Predictions. Elapsed: {:.2} secs",
        n_preds,
        duration_as_f64(&predict_start_time.elapsed()),
    );
    for pred in predictions.iter_mut() {
        *pred /= n_preds as f64;
    }

    let yhat_train: Vec<f64> = (0..train.features.n_rows())
        .map(|i| gbt.predict(&train.features.row(i)))
        .collect();
    println!("RMSE train {:.8}", rmse(&train.target, &yhat_train));

    let yhat_test: Vec<f64> = (0..test.features.n_rows())
        .map(|i| gbt.predict(&test.features.row(i)))
        .collect();
    println!("RMSE Test {:.8}", rmse(&test.target, &yhat_test));

    println!("Serializing model to example_boost.json");
    let serialized: String = serde_json::to_string(&gbt)?;
    let mut file = File::create("example_boost.json")?;
    file.write_all(serialized.as_bytes())?;

    println!("Writing predictions to example_boost.csv");
    let file = File::create("example_boost.csv")?;
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
