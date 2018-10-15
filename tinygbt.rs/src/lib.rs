#![feature(duration_as_u128)]
#![feature(plugin, custom_attribute)]

extern crate core;
extern crate ord_subset;
extern crate rand;

mod matrix;

use ord_subset::OrdSubsetSliceExt;
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
    fn _calc_split_gain(g: f64, h: f64, g_l: f64, h_l: f64, g_r: f64, h_r: f64, lambd: f64) -> f64 {
        fn calc_term(g: f64, h: f64, lambd: f64) -> f64 {
            g.powi(2) / (h + lambd)
        }
        calc_term(g_l, h_l, lambd) + calc_term(g_r, h_r, lambd) - calc_term(g, h, lambd)
    }

    /// Calculate the optimal weight of this leaf node.
    /// (Refer to Eq5 of Reference[1])
    fn _calc_leaf_weight(grad: &[f64], hessian: &[f64], lambda: f64, indices: &[usize]) -> f64 {
        return sum_indices(grad, indices) / (sum_indices(hessian, indices) + lambda);
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
        let mut best_gain = -::std::f64::INFINITY;
        let mut best_feature_id = None;
        let mut best_val = 0.;
        let mut best_left_instance_ids = None;
        let mut best_right_instance_ids = None;

        for feature_id in 0..nfeatures {
            let mut grad_left = 0.;
            let mut hessian_left = 0.;

            // sorted_instance_ids = instances[:, feature_id].argsort()
            let mut sorted_instance_ids: Vec<usize> = indices.clone().to_vec();
            sorted_instance_ids.sort_by_key(|&row_id| train.sorted_features[(row_id, feature_id)]);

            for (j, &nrow) in sorted_instance_ids.iter().enumerate() {
                grad_left += train.grad[nrow];
                hessian_left += train.hessian[nrow];
                let grad_right = sum_grad - grad_left;
                let hessian_right = sum_hessian - hessian_left;
                let current_gain = Self::_calc_split_gain(
                    sum_grad,
                    sum_hessian,
                    grad_left,
                    hessian_left,
                    grad_right,
                    hessian_right,
                    param.lambda,
                );

                if current_gain > best_gain {
                    best_gain = current_gain;
                    best_feature_id = Some(feature_id);
                    best_val = train.features[(nrow, feature_id)];
                    let (left_ids, right_ids) = sorted_instance_ids.split_at(j + 1);
                    best_left_instance_ids = Some(left_ids.to_vec());
                    best_right_instance_ids = Some(right_ids.to_vec());
                }
            }
        }

        let best_feature_id = best_feature_id.expect("best_feature_id");
        let best_left_instance_ids = best_left_instance_ids.expect("best_left_instance_ids");
        let best_right_instance_ids = best_right_instance_ids.expect("best_right_instance_ids");

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
}

pub struct GBT {
    models: Vec<Node>,
    params: Params,
    best_iteration: usize,
}

impl GBT {
    fn _calc_training_data_scores(&self, train_set: &Dataset, models: &[Node]) -> Option<Vec<f64>> {
        if models.len() == 0 {
            return None;
        }
        let n_rows = train_set.features.n_rows();
        let mut scores = Vec::with_capacity(n_rows);
        for i in 0..n_rows {
            scores.push(self._predict(&train_set.row(i), models));
        }
        Some(scores)
    }

    fn _calc_l2_gradient(
        &self,
        train_set: &Dataset,
        scores: Option<Vec<f64>>,
    ) -> (Vec<f64>, Vec<f64>) {
        let labels = &train_set.target;
        let hessian: Vec<f64> = (0..labels.len()).map(|_| 2.).collect();
        if let Some(scores) = scores {
            // println!("labels {}", labels.len());
            // println!("scores {}", scores.len());
            let grad = (0..labels.len())
                .map(|i| 2. * (labels[i] - scores[i]))
                .collect();
            return (grad, hessian);
        } else {
            let grad = (0..labels.len()).map(|_| random()).collect();
            //let grad = np.random.uniform(size = len(labels))
            (grad, hessian)
        }
    }

    fn _calc_l2_loss(&self, data_set: &Dataset, models: &[Node]) -> f64 {
        let mut errors = Vec::new();
        for (n_row, &target) in data_set.target.iter().enumerate() {
            let diff = target - self._predict(&data_set.row(n_row), &models);
            errors.push(diff.powi(2));
        }
        return mean(&errors);
    }

    /// For now, only L2 loss is supported
    fn _calc_loss(&self, data_set: &Dataset, models: &[Node]) -> f64 {
        self._calc_l2_loss(&data_set, models)
    }

    /// For now, only L2 loss is supported
    fn _calc_gradient(&self, data_set: &Dataset, scores: Option<Vec<f64>>) -> (Vec<f64>, Vec<f64>) {
        self._calc_l2_gradient(&data_set, scores)
    }

    fn prepare_train<'a>(
        &self,
        train_set: &'a Dataset,
        sorted_features: &'a ColumnMajorMatrix<usize>,
    ) -> TrainDataSet<'a> {
        let scores = self._calc_training_data_scores(train_set, &self.models);
        let (grad, hessian) = self._calc_gradient(train_set, scores);

        TrainDataSet {
            features: &train_set.features,
            target: &train_set.target,
            sorted_features,
            grad,
            hessian,
        }
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
        for iter_cnt in 0..(num_boost_round) {
            let iter_start_time = Instant::now();
            let train = self.prepare_train(&train_set, &sorted_features);
            let learner = self._build_learner(&train, shrinkage_rate);
            if iter_cnt > 0 {
                shrinkage_rate *= self.params.learning_rate;
            }
            self.models.push(learner);
            let train_loss = self._calc_loss(&train_set, &self.models);
            let val_loss = valid_set.map(|s| self._calc_loss(s, &self.models));
            let val_loss_str = val_loss
                .map(|e| format!("{:.10}", e))
                .unwrap_or("-".to_string());
            println!(
                "Iter {}, Train's L2: {:.10}, Valid's L2: {}, Elapsed: {:.2} secs for {} models",
                iter_cnt,
                train_loss,
                val_loss_str,
                iter_start_time.elapsed().as_nanos() as f64 / 1_000_000_000.,
                self.models.len(),
            );

            if let Some(val_loss_) = val_loss {
                if let Some(best_val_loss_) = best_val_loss {
                    if val_loss_ < best_val_loss_ || iter_cnt == 0 {
                        best_val_loss = val_loss;
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
            }
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
