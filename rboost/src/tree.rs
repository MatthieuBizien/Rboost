use crate::{Dataset, Params, StridedVecView, TrainDataSet};
use ord_subset::OrdSubsetIterExt;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use std::f64::INFINITY;

fn sum_indices(v: &[f64], indices: &[usize]) -> f64 {
    let mut o = 0.;
    for &i in indices {
        o += v[i];
    }
    o
}

pub(crate) struct SplitNode {
    left_child: Box<Node>,
    right_child: Box<Node>,
    split_feature_id: usize,
    split_val: f64,
}

pub(crate) struct LeafNode {
    val: f64,
}

pub(crate) enum Node {
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

    fn calc_gain_direct(
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

    fn calc_gain_bins(
        train: &TrainDataSet,
        indices: &[usize],
        sum_grad: f64,
        sum_hessian: f64,
        param: &Params,
        feature_id: usize,
    ) -> (usize, f64, f64, Vec<usize>, Vec<usize>) {
        let n_bin = train.n_bins[feature_id];
        let mut grads: Vec<_> = (0..n_bin).map(|_| 0.).collect();
        let mut hessians: Vec<_> = (0..n_bin).map(|_| 0.).collect();

        for &i in indices {
            let bin = train.bins[(i, feature_id)] as usize;
            grads[bin] += train.grad[i];
            hessians[bin] += train.hessian[i];
        }

        // We initialize at the first value
        let mut grad_left = grads[0];
        let mut hessian_left = hessians[0];
        let mut best_gain =
            Self::_calc_split_gain(sum_grad, sum_hessian, grad_left, hessian_left, param.lambda);
        let mut best_bin = 0;

        for bin in 1..n_bin {
            grad_left += grads[bin];
            hessian_left += hessians[bin];

            let current_gain = Self::_calc_split_gain(
                sum_grad,
                sum_hessian,
                grad_left,
                hessian_left,
                param.lambda,
            );

            if current_gain > best_gain {
                best_gain = current_gain;
                best_bin = bin;
            }
        }

        let mut left_ids = Vec::new();
        let mut right_ids = Vec::new();
        let mut best_val = INFINITY;
        for &i in indices {
            let bin = train.bins[(i, feature_id)] as usize;
            if bin < best_bin {
                left_ids.push(i);
            } else {
                right_ids.push(i);
                best_val = best_val.min(train.features[(i, feature_id)]);
            }
        }
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
    pub(crate) fn build(
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

        let get_results_bins = |feature_id| {
            Self::calc_gain_bins(&train, indices, sum_grad, sum_hessian, &param, feature_id)
        };
        let get_results_direct = |feature_id| {
            Self::calc_gain_direct(&train, indices, sum_grad, sum_hessian, &param, feature_id)
        };
        let results: Vec<_> = if param.n_bins > 0 {
            (0..nfeatures)
                .into_par_iter()
                .map(get_results_bins)
                .collect()
        } else {
            (0..nfeatures)
                .into_par_iter()
                .map(get_results_direct)
                .collect()
        };

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
