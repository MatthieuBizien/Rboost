use crate::losses::Loss;
use crate::{prod_vec, ColumnMajorMatrix};
use failure::Error;
use ordered_float::OrderedFloat;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use std::f64::INFINITY;
use std::ops::Deref;

// TODO use NonMaxX
type BinType = u32;

/// Util for parsing a CSV without headers into a dataset.
///
/// The first column of the CSV must be the target.
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

/// Store the raw data.
pub struct Dataset {
    /// Predictor for the learning
    pub features: ColumnMajorMatrix<f64>,
    /// Target, used for the learning
    pub target: Vec<f64>,
}

impl Dataset {
    /// Rank per columns: the smallest value will have the rank 1,
    /// two equals values will have the same rank.
    /// NAN values will have the rank 0.
    fn rank_features(&self) -> ColumnMajorMatrix<usize> {
        let columns = self
            .features
            .columns()
            .map(|column| {
                // Give the position in the column of the indices

                // First we sort the index according to the positions
                let mut sorted_indices: Vec<usize> = (0..column.len()).collect();
                sorted_indices.sort_by_key(|&row_id| OrderedFloat::from(column[row_id]));

                // Then we create the histogram of the features
                let mut w: Vec<usize> = (0..column.len()).map(|_| 0).collect();
                let nan_value = 0;
                let mut current_order = 1;
                let mut current_val = column[sorted_indices[0]];

                for idx in sorted_indices {
                    let val = column[idx];
                    if val.is_nan() {
                        w[idx] = nan_value;
                    } else {
                        if val != current_val {
                            if !current_val.is_nan() {
                                current_order += 1;
                            }
                            current_val = val;
                        }
                        w[idx] = current_order;
                    }
                }

                for &e in &w {
                    assert_ne!(e, 0);
                }
                w
            }).collect();
        ColumnMajorMatrix::from_columns(columns)
    }

    /// Bin values: the smallest value will have the bin 0, the biggest n_bins.
    /// NAN values will be None.
    /// Vec<usize> is the effective number of bins we have at the end.
    fn bin_features(
        features_rank: &ColumnMajorMatrix<usize>,
        n_bins: usize,
    ) -> (ColumnMajorMatrix<Option<BinType>>, Vec<usize>) {
        let x: Vec<_> = features_rank
            .columns()
            .map(|column| {
                let max: usize = 1 + *column.iter().max().expect("no data in col");
                let n_bins: usize = max.min(n_bins);
                assert!(n_bins < (BinType::max_value()) as usize);
                // Then we bins the features
                let bins: Vec<_> = column
                    .iter()
                    .map(|&e| Some((e * n_bins / max) as BinType))
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

    /// Pre-compute the thresholds when we split between two bins.
    fn get_threshold_between_bins(
        values: &[f64],
        bins: &[Option<BinType>],
        n_bin: usize,
    ) -> Vec<f64> {
        if n_bin == 0 {
            return Vec::new();
        }
        let mut min_vals = vec![INFINITY; n_bin];
        let mut max_vals = vec![-INFINITY; n_bin];
        for (&val, &bin) in values.iter().zip(bins.iter()) {
            let bin = match bin {
                None => continue,
                Some(e) => e as usize,
            };
            min_vals[bin] = min_vals[bin].min(val);
            max_vals[bin] = max_vals[bin].max(val);
        }
        // TODO what happens if a bin is empty?
        max_vals
            .into_iter()
            .zip(min_vals.into_iter().skip(1))
            .map(|(a, b)| (a / 2. + b / 2.))
            .collect()
    }

    /// Prepare the dataset for the training.
    /// * `n_bins` - Number of bins we want to use. Set it to 0 for exact training
    ///     Exact training is slower and more prone to over-fit.
    pub fn as_prepared_data(&self, n_bins: usize) -> PreparedDataset {
        let features_rank = self.rank_features();
        let (bins, n_bins) = Dataset::bin_features(&features_rank, n_bins);

        let threshold_vals: Vec<_> = self
            .features
            .columns()
            .zip(bins.columns())
            .zip(n_bins.iter())
            .collect();

        let threshold_vals = threshold_vals
            .into_par_iter()
            .map(|((values, bins), &n_bin)| Self::get_threshold_between_bins(values, bins, n_bin))
            .collect();

        PreparedDataset {
            features: &self.features,
            target: &self.target,
            features_rank,
            bins,
            n_bins,
            threshold_vals,
        }
    }
}

/// Dataset pre-computed for the training.
pub struct PreparedDataset<'a> {
    pub(crate) features: &'a ColumnMajorMatrix<f64>,
    pub(crate) target: &'a Vec<f64>,
    // Rank inside the dataset of a feature. Can contains duplicates if the values are equals.
    pub(crate) features_rank: ColumnMajorMatrix<usize>,
    pub(crate) bins: ColumnMajorMatrix<Option<BinType>>,
    pub(crate) n_bins: Vec<usize>,
    pub(crate) threshold_vals: Vec<Vec<f64>>,
}

impl<'a> PreparedDataset<'a> {
    pub(crate) fn as_train_data(&'a self, loss: &impl Loss) -> TrainDataset<'a> {
        let zero_vec: Vec<_> = self.target.iter().map(|_| 0.).collect();
        let weights: Vec<_> = self.target.iter().map(|_| 1.).collect();
        let columns: Vec<_> = (0..self.features.n_cols()).collect();
        let mut train = TrainDataset {
            grad: zero_vec.clone(),
            hessian: zero_vec.clone(),
            columns,
            data: self,
        };
        train.update_grad_hessian(loss, &zero_vec, &weights);
        train
    }
}

pub(crate) struct TrainDataset<'a> {
    pub(crate) grad: Vec<f64>,
    pub(crate) hessian: Vec<f64>,
    // Columns that we want to train on
    pub(crate) columns: Vec<usize>,
    pub(crate) data: &'a PreparedDataset<'a>,
}

// With Deref we can use train_data_set.X if X is an attribute of PreparedDataset
impl<'a> Deref for TrainDataset<'a> {
    type Target = PreparedDataset<'a>;
    fn deref(&self) -> &Self::Target {
        self.data
    }
}

impl<'a> TrainDataset<'a> {
    pub(crate) fn update_grad_hessian(
        &mut self,
        loss: &impl Loss,
        predictions: &[f64],
        sample_weights: &[f64],
    ) {
        assert_eq!(predictions.len(), sample_weights.len());
        let (grad, hessian) = loss.calc_gradient_hessian(&self.target, &predictions);
        self.grad = prod_vec(&grad, sample_weights);
        self.hessian = prod_vec(&hessian, sample_weights);
    }
}
