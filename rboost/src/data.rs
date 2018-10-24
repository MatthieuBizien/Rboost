use crate::losses::Loss;
use crate::{ColumnMajorMatrix, StridedVecView};
use ord_subset::OrdSubsetSliceExt;

type BinType = u32;

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
}

pub(crate) struct TrainDataSet<'a> {
    pub features: &'a ColumnMajorMatrix<f64>,
    pub target: &'a Vec<f64>,
    pub grad: Vec<f64>,
    pub hessian: Vec<f64>,
    pub sorted_features: ColumnMajorMatrix<usize>,
    pub bins: ColumnMajorMatrix<BinType>,
    pub n_bins: Vec<usize>,
}

impl<'a> TrainDataSet<'a> {
    pub(crate) fn update_grad_hessian(&mut self, loss: &impl Loss, predictions: &[f64]) {
        let (grad, hessian) = loss.calc_gradient(&self.target, &predictions);
        self.grad = grad;
        self.hessian = hessian;
    }
}
