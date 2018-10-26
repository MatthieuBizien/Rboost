use crate::losses::Loss;
use crate::{ColumnMajorMatrix, StridedVecView};
use failure::Error;
use ord_subset::OrdSubsetSliceExtMut;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use std::f64::INFINITY;

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

    pub(crate) fn as_train_data(&self, n_bins: usize) -> TrainDataSet {
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

        TrainDataSet {
            features: &self.features,
            target: &self.target,
            grad: Vec::new(),
            hessian: Vec::new(),
            sorted_features,
            bins,
            n_bins,
            threshold_vals,
        }
    }
}

pub(crate) struct TrainDataSet<'a> {
    pub features: &'a ColumnMajorMatrix<f64>,
    pub target: &'a Vec<f64>,
    pub grad: Vec<f64>,
    pub hessian: Vec<f64>,
    pub sorted_features: ColumnMajorMatrix<usize>,
    pub bins: ColumnMajorMatrix<BinType>,
    pub n_bins: Vec<usize>,
    pub threshold_vals: Vec<Vec<f64>>,
}

impl<'a> TrainDataSet<'a> {
    pub(crate) fn update_grad_hessian(&mut self, loss: &impl Loss, predictions: &[f64]) {
        let (grad, hessian) = loss.calc_gradient(&self.target, &predictions);
        self.grad = grad;
        self.hessian = hessian;
    }
}
