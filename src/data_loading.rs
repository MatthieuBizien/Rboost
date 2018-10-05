use std::fs::File;
use std::io::{BufRead, BufReader};
use std::option::Option;

use data::Matrix;

type Float = f32;
type Target = f64;

pub fn read_csv(filename: &str) -> Matrix {
    let f = File::open(filename).expect("file not found");
    let mut values = Vec::new();
    let mut labels = Vec::new();
    for line in BufReader::new(f).lines() {
        let line = line.expect("IO Error");
        let mut elements = line.split(",").map(|x| x.parse().expect("Not a float"));
        let element: Option<Target> = elements.next();
        match element {
            None => continue,
            Some(element) => {
                labels.push(element as Target);
                values.push(elements.map(|x| x as Float).collect());
            }
        }
    }

    Matrix::new(
        values.into_iter().take(10).collect(),
        labels.into_iter().take(10).collect(),
    )
}
