extern crate cpuprofiler;
extern crate failure;
extern crate rboost;

use cpuprofiler::PROFILER;
use failure::Error;
use rboost::{ColumnMajorMatrix, Dataset, Params, GBT};

fn parse_tsv(data: &str) -> Result<Dataset, Error> {
    let mut target: Vec<f64> = Vec::new();
    let mut features: Vec<Vec<f64>> = Vec::new();
    for l in data.split("\n") {
        if l.len() == 0 {
            continue;
        }
        let mut items = l.split("\t").into_iter();
        target.push(items.next().expect("first item").parse()?);
        features.push(items.map(|e| e.parse().unwrap()).collect());
    }
    let features = ColumnMajorMatrix::from_rows(features);

    Ok(Dataset { features, target })
}

fn rmse(target: &[f64], yhat: &[f64]) -> f64 {
    let rmse: f64 = yhat
        .iter()
        .zip(target.iter())
        .map(|(&a, &b)| (a - b).powi(2))
        .sum();
    (rmse / target.len() as f64).sqrt()
}

fn main() {
    let train = include_str!("../data/regression.train");
    let train = parse_tsv(train).expect("Train data");
    let test = include_str!("../data/regression.test");
    let test = parse_tsv(test).expect("Train data");

    let params = Params {
        gamma: 0.,
        lambda: 1.,
        learning_rate: 0.8,
        max_depth: 3,
        min_split_gain: 0.1,
        n_bins: 255,
    };
    PROFILER.lock().unwrap().start("./my-prof.profile").unwrap();
    let gbt = GBT::build(&params, &train, 100, Some(&test), 100);
    PROFILER.lock().unwrap().stop().unwrap();
    let yhat: Vec<f64> = (0..train.features.n_rows())
        .map(|i| gbt.predict(&train.row(i)))
        .collect();
    println!("RMSE train {:.8}", rmse(&train.target, &yhat));
    let yhat: Vec<f64> = (0..test.features.n_rows())
        .map(|i| gbt.predict(&test.row(i)))
        .collect();
    println!("RMSE Test {:.8}", rmse(&test.target, &yhat));
}
