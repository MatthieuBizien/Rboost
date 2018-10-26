extern crate cpuprofiler;
extern crate csv;
extern crate failure;
extern crate rboost;
extern crate serde_json;

use cpuprofiler::PROFILER;
use rboost::{parse_csv, rmse, Params, RegLoss, GBT};
use std::fs::File;
use std::io::Write;

fn main() {
    let train = include_str!("../data/regression.train");
    let train = parse_csv(train, "\t").expect("Train data");
    let test = include_str!("../data/regression.test");
    let test = parse_csv(test, "\t").expect("Train data");

    let params = Params {
        gamma: 0.,
        lambda: 1.,
        learning_rate: 0.99,
        max_depth: 3,
        min_split_gain: 0.1,
        n_bins: 128,
    };
    println!("Params {:?}", params);
    println!("Profiling to example1.profile");
    PROFILER
        .lock()
        .unwrap()
        .start("./example1.profile")
        .unwrap();
    let gbt = GBT::build(&params, &train, 1000, Some(&test), 1000, RegLoss::default());
    PROFILER.lock().unwrap().stop().unwrap();

    let yhat_train: Vec<f64> = (0..train.features.n_rows())
        .map(|i| gbt.predict(&train.row(i)))
        .collect();
    println!("RMSE train {:.8}", rmse(&train.target, &yhat_train));

    let yhat_test: Vec<f64> = (0..test.features.n_rows())
        .map(|i| gbt.predict(&test.row(i)))
        .collect();
    println!("RMSE Test {:.8}", rmse(&test.target, &yhat_test));

    println!("Serializing model to example1.json");
    let serialized: String = serde_json::to_string(&gbt).expect("Error on JSON serialization");
    let mut file = File::create("example1.json").expect("Error on file creation");
    file.write_all(serialized.as_bytes())
        .expect("Error on writing of the JSON");

    println!("Writing predictions to example1.csv");
    let file = File::create("example1.csv").expect("Error on file creation");
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
