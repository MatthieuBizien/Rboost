extern crate failure;
extern crate tinygbt;

use failure::Error;
use nalgebra::DMatrix;
use tinygbt::{Dataset, Params, GBT};

fn parse_tsv(data: &str) -> Result<Dataset, Error> {
    let mut target: Vec<f64> = Vec::new();
    let mut features: Vec<f64> = Vec::new();
    let mut ncols = 0;
    let mut nrows = 0;
    for (i, l) in data.split("\n").enumerate() {
        if l.len() == 0 {
            continue;
        }
        let mut items = l.split("\t").into_iter();
        target.push(items.next().expect("first item").parse()?);
        let current: Vec<f64> = items.map(|e| e.parse().unwrap()).collect();
        if i == 0 {
            ncols = current.len();
        } else {
            assert_eq!(current.len(), ncols);
        }
        features.extend(current);
        nrows += 1;
    }
    let features = DMatrix::from_row_slice(nrows, ncols, &features);

    Ok(Dataset { features, target })
}

fn main() {
    let train = include_str!("../../tinygbt/data/regression.train");
    let train = parse_tsv(train).expect("Train data");
    let test = include_str!("../../tinygbt/data/regression.test");
    let test = parse_tsv(test).expect("Train data");

    let params = Params {
        gamma: 0.,
        lambda: 1.,
        learning_rate: 0.3,
        max_depth: 5,
        min_split_gain: 0.1,
    };

    let gbt = GBT::build(&params, &train, 20, Some(&test), 5);
    let yhat: Vec<f64> = (0..test.features.nrows())
        .map(|i| gbt.predict(&test.row(i)))
        .collect();
    let rmse: f64 = yhat
        .iter()
        .zip(test.target.iter())
        .map(|(&a, &b)| (a - b).powi(2))
        .sum();
    let rmse = (rmse / test.target.len() as f64).sqrt();
    println!("RMSE Test {:.8}", rmse);
}
