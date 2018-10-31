use crate::losses::Loss;
use crate::{prod_vec, ColumnMajorMatrix, StridedVecView};
use failure::Error;
use ord_subset::OrdSubsetSliceExtMut;
use rand::prelude::Rng;
use rand::seq::sample_slice;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use std::f64::INFINITY;
use std::ops::Deref;

type BinType = u32;

pub fn parse_csv(data: &str, sep: &str) -> Result<Dataset, Error> {
    let mut target: Vec<f64> = Vec::new();
    let mut features: Vec<Vec<f64>> = Vec::new();
    for l in data.split("\n") {
        if l.len() == 0 {
            continue;
        }
        let mut items = l.split(sep).into_iter();
        target.push(items.next().expect("first item").parse()?);
        features.push(items.map(|e| e.parse().unwrap()).collect());
    }
    let features = ColumnMajorMatrix::from_rows(features);

    Ok(Dataset { features, target })
}

pub struct Dataset {
    /// Order is inverted: rows are features, columns are observation
    pub features: ColumnMajorMatrix<f64>,
    pub target: Vec<f64>,
}

impl Dataset {
    pub fn row(&self, n_row: usize) -> StridedVecView<f64> {
        self.features.row(n_row)
    }

    pub fn sort_features(&self) -> ColumnMajorMatrix<usize> {
        let columns = self
            .features
            .columns()
            .map(|column| {
                // Give the position in the column of the indices

                // First we sort the index according to the positions
                let mut sorted_indices: Vec<usize> = (0..column.len()).collect();
                sorted_indices.ord_subset_sort_by_key(|&row_id| column[row_id]);

                // Then we create the histogram of the features
                let mut w: Vec<usize> = (0..column.len()).map(|_| 0).collect();
                let mut current_order = 0;
                let mut current_val = column[sorted_indices[0]];
                for idx in sorted_indices.into_iter().skip(1) {
                    let val = column[idx];
                    if val != current_val {
                        current_order += 1;
                        current_val = val;
                    }
                    w[idx] = current_order;
                }
                w
            }).collect();
        ColumnMajorMatrix::from_columns(columns)
    }

    pub fn bin_features(
        sorted_features: &ColumnMajorMatrix<usize>,
        n_bins: usize,
    ) -> (ColumnMajorMatrix<BinType>, Vec<usize>) {
        let x: Vec<_> = sorted_features
            .columns()
            .map(|column| {
                let max: usize = 1 + *column.iter().max().expect("no data in col");
                let n_bins: usize = max.min(n_bins);
                assert!(n_bins < (BinType::max_value()) as usize);
                // Then we bins the features
                let bins: Vec<BinType> = column
                    .iter()
                    .map(|&e| (e * n_bins / max) as BinType)
                    .collect();
                (bins, n_bins)
            }).collect();
        let mut columns = Vec::with_capacity(x.len());
        let mut n_bins = Vec::with_capacity(x.len());
        for (col, n_bin) in x.into_iter() {
            columns.push(col);
            n_bins.push(n_bin);
        }
        let columns = ColumnMajorMatrix::from_columns(columns);
        (columns, n_bins)
    }

    fn get_threshold_vals(values: &[f64], bins: &[BinType], n_bin: usize) -> Vec<f64> {
        if n_bin == 0 {
            return Vec::new();
        }
        let mut min_vals = vec![INFINITY; n_bin];
        let mut max_vals = vec![-INFINITY; n_bin];
        for (&val, &bin) in values.iter().zip(bins.iter()) {
            let bin = bin as usize;
            min_vals[bin] = min_vals[bin].min(val);
            max_vals[bin] = max_vals[bin].max(val);
        }
        max_vals
            .into_iter()
            .zip(min_vals.into_iter().skip(1))
            .map(|(a, b)| (a / 2. + b / 2.))
            .collect()
    }

    pub fn as_prepared_data(&self, n_bins: usize) -> PreparedDataSet {
        let sorted_features = self.sort_features();
        let (bins, n_bins) = Dataset::bin_features(&sorted_features, n_bins);

        let threshold_vals: Vec<_> = self
            .features
            .columns()
            .zip(bins.columns())
            .zip(n_bins.iter())
            .collect();

        let threshold_vals = threshold_vals
            .into_par_iter()
            .map(|((values, bins), &n_bin)| Self::get_threshold_vals(values, bins, n_bin))
            .collect();

        PreparedDataSet {
            features: &self.features,
            target: &self.target,
            sorted_features,
            bins,
            n_bins,
            threshold_vals,
        }
    }
}

pub struct PreparedDataSet<'a> {
    pub(crate) features: &'a ColumnMajorMatrix<f64>,
    pub target: &'a Vec<f64>,
    pub(crate) sorted_features: ColumnMajorMatrix<usize>,
    pub(crate) bins: ColumnMajorMatrix<BinType>,
    pub(crate) n_bins: Vec<usize>,
    pub(crate) threshold_vals: Vec<Vec<f64>>,
}

impl<'a> PreparedDataSet<'a> {
    pub(crate) fn as_train_data(&'a self, loss: &impl Loss) -> TrainDataSet<'a> {
        let zero_vec: Vec<_> = self.target.iter().map(|_| 0.).collect();
        let weights: Vec<_> = self.target.iter().map(|_| 1.).collect();
        let columns: Vec<_> = (0..self.features.n_cols()).collect();
        let mut train = TrainDataSet {
            grad: zero_vec.clone(),
            hessian: zero_vec.clone(),
            columns,
            data: self,
        };
        train.update_grad_hessian(loss, &zero_vec, &weights);
        train
    }
}

pub(crate) struct TrainDataSet<'a> {
    pub(crate) grad: Vec<f64>,
    pub(crate) hessian: Vec<f64>,
    pub(crate) columns: Vec<usize>,
    pub(crate) data: &'a PreparedDataSet<'a>,
}

// With Deref we can use train_data_set.X if X is an attribute of PreparedDataSet
impl<'a> Deref for TrainDataSet<'a> {
    type Target = PreparedDataSet<'a>;
    fn deref(&self) -> &Self::Target {
        self.data
    }
}

impl<'a> TrainDataSet<'a> {
    pub(crate) fn update_grad_hessian(
        &mut self,
        loss: &impl Loss,
        predictions: &[f64],
        sample_weights: &[f64],
    ) {
        assert_eq!(predictions.len(), sample_weights.len());
        let (grad, hessian) = loss.calc_gradient(&self.target, &predictions);
        self.grad = prod_vec(&grad, sample_weights);
        self.hessian = prod_vec(&hessian, sample_weights);
    }

    pub(crate) fn update_columns(
        &mut self,
        colsample_bytree: f64,
        reset: bool,
        rng: &mut impl Rng,
    ) {
        if reset {
            self.columns = (0..self.features.n_cols()).collect();
        }
        let n_cols_f64 = colsample_bytree * (self.features.n_cols() as f64);
        let mut n_cols = n_cols_f64.floor() as usize;
        // Randomly select N or N+1 column proportionally to the difference
        if (n_cols_f64 - n_cols as f64) > rng.gen() {
            n_cols += 1;
        }
        self.columns = sample_slice(rng, self.columns.as_slice(), n_cols);
    }
}
