#![feature(duration_as_u128)]

extern crate cpuprofiler;
extern crate csv;
extern crate failure;
extern crate rboost;
extern crate serde_json;

use cpuprofiler::PROFILER;
use rand::prelude::{SeedableRng, SmallRng};
use rboost::{parse_csv, rmse, Dart, DartParams, RegLoss, TreeParams};
use std::fs::File;
use std::io::Write;
use std::time::Instant;

fn main() {
    let train = include_str!("../data/regression.train");
    let train = parse_csv(train, "\t").expect("Train data");
    let test = include_str!("../data/regression.test");
    let test = parse_csv(test, "\t").expect("Train data");

    let seed = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]; // byte array
    let mut rng = SmallRng::from_seed(seed);

    let booster_params = DartParams {
        colsample_bytree: 0.95,
        dropout_rate: 0.97,
    };
    let tree_params = TreeParams {
        gamma: 1.,
        lambda: 10.,
        max_depth: 3,
        min_split_gain: 1.,
    };
    let n_bins = 2048;
    println!(
        "Params booster={:?} tree{:?} n_bins={}",
        booster_params, tree_params, n_bins
    );
    println!("Profiling to example_dart.profile");
    PROFILER
        .lock()
        .unwrap()
        .start("./example_dart.profile")
        .unwrap();
    let gbt = Dart::build(
        &booster_params,
        &tree_params,
        &mut train.as_prepared_data(n_bins),
        1000,
        Some(&test),
        1000,
        RegLoss::default(),
        &mut rng,
    );
    PROFILER.lock().unwrap().stop().unwrap();

    let n_preds = 10;
    let predict_start_time = Instant::now();
    let mut predictions: Vec<_> = train.target.iter().map(|_| 0.).collect();
    for _ in 0..n_preds {
        for (i, pred) in predictions.iter_mut().enumerate() {
            *pred += gbt.predict(&train.features.row(i));
        }
    }
    println!(
        "{} Predictions. Elapsed: {:.2} secs",
        n_preds,
        predict_start_time.elapsed().as_nanos() as f64 / 1_000_000_000.
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

    println!("Serializing model to example_dart.json");
    let serialized: String = serde_json::to_string(&gbt).expect("Error on JSON serialization");
    let mut file = File::create("example_dart.json").expect("Error on file creation");
    file.write_all(serialized.as_bytes())
        .expect("Error on writing of the JSON");

    println!("Writing predictions to example_dart.csv");
    let file = File::create("example_dart.csv").expect("Error on file creation");
    let mut wtr = csv::Writer::from_writer(file);
    wtr.write_record(&["dataset", "true_val", "yhat"])
        .expect("Error on csv writing");
    for (true_val, yhat) in train.target.iter().zip(yhat_train.iter()) {
        wtr.write_record(&["train", &true_val.to_string(), &yhat.to_string()])
            .expect("Error on csv writing");
    }
    for (true_val, yhat) in test.target.iter().zip(yhat_test.iter()) {
        wtr.write_record(&["test", &true_val.to_string(), &yhat.to_string()])
            .expect("Error on csv writing");
    }
    wtr.flush().expect("Error on CSV flushing");
}
