#![feature(duration_as_u128)]
#![feature(plugin, custom_attribute)]
#![feature(inner_deref)]

extern crate core;
extern crate ord_subset;
extern crate rand;
extern crate rayon;

mod matrix;

use ord_subset::{OrdSubsetIterExt, OrdSubsetSliceExt};
use rayon::prelude::*;
use std::time::Instant;

use rand::random;

pub use crate::matrix::{ColumnMajorMatrix, StridedVecView};

fn sum(v: &[f64]) -> f64 {
    let mut o = 0.;
    for e in v.iter() {
        o += *e;
    }
    o
}

fn sum_indices(v: &[f64], indices: &[usize]) -> f64 {
    let mut o = 0.;
    for &i in indices {
        o += v[i];
    }
    o
}

fn mean(v: &[f64]) -> f64 {
    sum(&v) / (v.len() as f64)
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
}

struct TrainDataSet<'a> {
    pub features: &'a ColumnMajorMatrix<f64>,
    pub sorted_features: &'a ColumnMajorMatrix<usize>,
    pub target: &'a Vec<f64>,
    pub grad: Vec<f64>,
    pub hessian: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct Params {
    pub gamma: f64,
    pub lambda: f64,
    pub learning_rate: f64,
    pub max_depth: usize,
    pub min_split_gain: f64,
}

struct SplitNode {
    left_child: Box<Node>,
    right_child: Box<Node>,
    split_feature_id: usize,
    split_val: f64,
}

struct LeafNode {
    val: f64,
}

enum Node {
    Split(SplitNode),
    Leaf(LeafNode),
}

impl Node {
    ///  Loss reduction
    /// (Refer to Eq7 of Reference[1])
    fn _calc_split_gain(g: f64, h: f64, g_l: f64, h_l: f64, lambd: f64) -> f64 {
        fn calc_term(g: f64, h: f64, lambd: f64) -> f64 {
            g.powi(2) / (h + lambd)
        }
        let g_r = g - g_l;
        let h_r = h - h_l;
        calc_term(g_l, h_l, lambd) + calc_term(g_r, h_r, lambd) - calc_term(g, h, lambd)
    }

    /// Calculate the optimal weight of this leaf node.
    /// (Refer to Eq5 of Reference[1])
    fn _calc_leaf_weight(grad: &[f64], hessian: &[f64], lambda: f64, indices: &[usize]) -> f64 {
        return sum_indices(grad, indices) / (sum_indices(hessian, indices) + lambda);
    }

    fn calc_gain(
        train: &TrainDataSet,
        indices: &[usize],
        sum_grad: f64,
        sum_hessian: f64,
        param: &Params,
        feature_id: usize,
    ) -> (usize, f64, f64, Vec<usize>, Vec<usize>) {
        // sorted_instance_ids = instances[:, feature_id].argsort()
        let mut sorted_instance_ids: Vec<usize> = indices.clone().to_vec();
        sorted_instance_ids
            .sort_unstable_by_key(|&row_id| train.sorted_features[(row_id, feature_id)]);

        // We initialize at the first value
        let mut grad_left = train.grad[sorted_instance_ids[0]];
        let mut hessian_left = train.hessian[sorted_instance_ids[0]];
        let mut best_gain =
            Self::_calc_split_gain(sum_grad, sum_hessian, grad_left, hessian_left, param.lambda);
        let mut best_idx = 0;
        let mut best_val = train.features[(0, feature_id)];
        let mut last_val = train.features[(0, feature_id)];

        for (j, &nrow) in sorted_instance_ids.iter().skip(1).enumerate() {
            grad_left += train.grad[nrow];
            hessian_left += train.hessian[nrow];

            // We can only split when the value change
            let val = train.features[(nrow, feature_id)];
            if val == last_val {
                continue;
            }
            last_val = val;

            let current_gain = Self::_calc_split_gain(
                sum_grad,
                sum_hessian,
                grad_left,
                hessian_left,
                param.lambda,
            );

            if current_gain > best_gain {
                best_gain = current_gain;
                best_idx = j;
                best_val = val;
            }
        }

        let (left_ids, right_ids) = sorted_instance_ids.split_at(best_idx + 1);
        (
            feature_id,
            best_val,
            best_gain,
            left_ids.to_vec(),
            right_ids.to_vec(),
        )
    }

    /// Exact Greedy Algorithm for Split Finding
    ///  (Refer to Algorithm1 of Reference[1])
    fn build(
        train: &TrainDataSet,
        indices: &[usize],
        shrinkage_rate: f64,
        depth: usize,
        param: &Params,
    ) -> Node {
        let nfeatures = train.features.n_cols() as usize;

        if depth > param.max_depth {
            let val = Node::_calc_leaf_weight(&train.grad, &train.hessian, param.lambda, indices)
                * shrinkage_rate;
            return Node::Leaf(LeafNode { val });
        }

        let sum_grad = sum_indices(&train.grad, indices);
        let sum_hessian = sum_indices(&train.hessian, indices);

        let results: Vec<_> = (0..nfeatures)
            .into_par_iter()
            .map(|feature_id| {
                Self::calc_gain(&train, indices, sum_grad, sum_hessian, &param, feature_id)
            }).collect();

        let (best_feature_id, best_val, best_gain, best_left_instance_ids, best_right_instance_ids) =
            results
                .into_iter()
                .ord_subset_max_by_key(|(_, _, gain, _, _)| gain.clone())
                .expect("Impossible to get the best gain for unknown reason");

        if best_gain < param.min_split_gain {
            let val = Node::_calc_leaf_weight(&train.grad, &train.hessian, param.lambda, indices)
                * shrinkage_rate;
            return Node::Leaf(LeafNode { val });
        }

        let get_child = |instance_ids: Vec<usize>| {
            Box::new(Self::build(
                &train,
                &instance_ids,
                shrinkage_rate,
                depth + 1,
                &param,
            ))
        };

        let left_child = get_child(best_left_instance_ids);
        let right_child = get_child(best_right_instance_ids);

        Node::Split(SplitNode {
            left_child,
            right_child,
            split_feature_id: best_feature_id,
            split_val: best_val,
        })
    }

    pub fn predict(&self, features: &StridedVecView<f64>) -> f64 {
        match &self {
            Node::Split(node) => {
                if features[node.split_feature_id] <= node.split_val {
                    node.left_child.predict(&features)
                } else {
                    node.right_child.predict(&features)
                }
            }
            Node::Leaf(node) => node.val,
        }
    }

    pub fn par_predict(&self, train_set: &Dataset) -> Vec<f64> {
        (0..train_set.target.len())
            .into_par_iter()
            .map(|i| {
                let row = train_set.features.row(i);
                self.predict(&row)
            }).collect()
    }
}

pub struct GBT {
    models: Vec<Node>,
    params: Params,
    best_iteration: usize,
}

impl GBT {
    fn _calc_l2_gradient(&self, target: &[f64], predictions: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let hessian: Vec<f64> = (0..target.len()).map(|_| 2.).collect();
        let grad = (0..target.len())
            .map(|i| 2. * (target[i] - predictions[i]))
            .collect();
        (grad, hessian)
    }

    fn _calc_l2_loss(&self, target: &[f64], predictions: &[f64]) -> f64 {
        let mut errors = Vec::new();
        for (n_row, &target) in target.iter().enumerate() {
            let diff = target - predictions[n_row];
            errors.push(diff.powi(2));
        }
        return mean(&errors);
    }

    /// For now, only L2 loss is supported
    fn _calc_loss(&self, target: &[f64], predictions: &[f64]) -> f64 {
        self._calc_l2_loss(target, predictions)
    }

    /// For now, only L2 loss is supported
    fn _calc_gradient(&self, target: &[f64], predictions: &[f64]) -> (Vec<f64>, Vec<f64>) {
        self._calc_l2_gradient(target, predictions)
    }

    fn _build_learner(&self, train: &TrainDataSet, shrinkage_rate: f64) -> Node {
        let depth = 0;
        let indices: Vec<usize> = (0..train.target.len()).collect();
        Node::build(&train, &indices, shrinkage_rate, depth, &self.params)
    }

    pub fn build(
        params: &Params,
        train_set: &Dataset,
        num_boost_round: usize,
        valid_set: Option<&Dataset>,
        early_stopping_rounds: usize,
    ) -> Self {
        let mut o = GBT {
            models: Vec::new(),
            params: (*params).clone(),
            best_iteration: 0,
        };
        o.train(train_set, num_boost_round, valid_set, early_stopping_rounds);
        o
    }

    fn train(
        &mut self,
        train_set: &Dataset,
        num_boost_round: usize,
        valid_set: Option<&Dataset>,
        early_stopping_rounds: usize,
    ) {
        // Check we have no NAN in input
        for &x in train_set.features.flat() {
            if x.is_nan() {
                panic!("Found NAN in the features")
            }
        }
        for &x in &train_set.target {
            if x.is_nan() {
                panic!("Found NAN in the features")
            }
        }

        let mut shrinkage_rate = 1.;
        let mut best_iteration = 0;
        let mut best_val_loss = None;
        let train_start_time = Instant::now();

        let sorted_features = train_set.sort_features();
        println!(
            "Features sorted in {:.03}s",
            train_start_time.elapsed().as_nanos() as f64 / 1_000_000_000.
        );

        println!(
            "Training until validation scores don't improve for {} rounds.",
            early_stopping_rounds
        );
        let mut train_scores: Vec<f64> = (0..train_set.target.len()).map(|_| 0.).collect();
        let mut val_scores: Option<Vec<f64>> =
            valid_set.map(|dataset| (0..dataset.target.len()).map(|_| 0.).collect());

        for iter_cnt in 0..(num_boost_round) {
            let iter_start_time = Instant::now();
            let (grad, hessian) = self._calc_gradient(&train_set.target, &train_scores);

            // TODO seems to work, but is it sound?
            let grad = if iter_cnt == 0 {
                grad.into_iter().map(|i| i * 0.5).collect()
            } else {
                grad
            };

            let train = TrainDataSet {
                features: &train_set.features,
                target: &train_set.target,
                sorted_features: &sorted_features,
                grad,
                hessian,
            };
            let learner = self._build_learner(&train, shrinkage_rate);
            if iter_cnt > 0 {
                shrinkage_rate *= self.params.learning_rate;
            }
            for (i, val) in learner.par_predict(&train_set).into_iter().enumerate() {
                train_scores[i] += val;
            }

            if let Some(valid_set) = valid_set {
                let val_scores = val_scores.as_mut().expect("No val score");
                for (i, val) in learner.par_predict(&valid_set).into_iter().enumerate() {
                    val_scores[i] += val
                }
                let val_loss = self._calc_loss(&valid_set.target, &val_scores);

                if let Some(best_val_loss_) = best_val_loss {
                    if val_loss < best_val_loss_ || iter_cnt == 0 {
                        best_val_loss = Some(val_loss);
                        best_iteration = iter_cnt;
                    }
                    if iter_cnt - best_iteration >= early_stopping_rounds {
                        println!("Early stopping, best iteration is:");
                        println!(
                            "Iter {:}, Train's L2: {:.10}",
                            best_iteration,
                            best_val_loss
                                .map(|e| format!("{:.10}", e))
                                .unwrap_or("-".to_string())
                        );
                        break;
                    }
                }
                self.models.push(learner);
            };
            /*
            println!(
                "Iter {}, Train's L2: {:.10}, Valid's L2: {}, Elapsed: {:.2} secs for {} models",
                iter_cnt,
                self._calc_loss(&train_set.target, &train_scores),
                valid_set.map(|s| self._calc_loss(&s.target, val_scores.deref().unwrap()))
                    .map(|val_loss| format!("{:.10}", val_loss))
                    .unwrap_or("-".to_string()),
                iter_start_time.elapsed().as_nanos() as f64 / 1_000_000_000.,
                self.models.len(),
            );*/
        }

        self.best_iteration = best_iteration;
        println!(
            "Training finished. Elapsed: {:.2} secs",
            train_start_time.elapsed().as_nanos() as f64 / 1_000_000_000.
        );
    }

    fn _predict(&self, features: &StridedVecView<f64>, models: &[Node]) -> f64 {
        let o: f64 = models.iter().map(|model| model.predict(features)).sum();
        if o.is_nan() {
            panic!("NAN in output of prediction");
        }
        o
    }

    pub fn predict(&self, features: &StridedVecView<f64>) -> f64 {
        self._predict(&features, &self.models)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
