#![feature(duration_as_u128)]

extern crate csv;
extern crate failure;
extern crate rboost;
extern crate serde_json;

#[macro_use]
extern crate criterion;

use criterion::{Bencher, Criterion};

use rboost::{parse_csv, Node, RegLoss, TreeParams};

fn bench_tree(b: &mut Bencher, n_bins: &&usize) {
    let train = include_str!("../data/regression.train");
    let train = parse_csv(train, "\t").expect("Train data");
    let train = train.as_prepared_data(**n_bins);
    let indices: Vec<_> = (0..train.features.n_rows()).collect();
    let loss = RegLoss::default();

    let large_cache = train.features.n_rows() * 100;

    b.iter_with_large_setup(
        || Vec::with_capacity(large_cache),
        |mut cache| {
            let tree_params = TreeParams {
                gamma: 1.,
                lambda: 10.,
                max_depth: 10,
                min_split_gain: 1.,
            };

            let mut predictions: Vec<_> = (0..train.features.n_rows()).map(|_| 0.).collect();
            Node::build(
                &train,
                &indices,
                &mut predictions,
                &tree_params,
                &mut cache,
                &loss,
            );
        },
    )
}

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function_over_inputs("tree", bench_tree, &[0, 256]);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
