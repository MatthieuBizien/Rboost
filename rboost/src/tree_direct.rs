use crate::{sum_indices, weighted_mean, LeafNode, Node, SplitNode, TrainDataSet, TreeParams};
use ord_subset::OrdSubsetIterExt;
//use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use std::f64::INFINITY;

/// Store the result of a successful split on a node
struct SplitResult {
    feature_id: usize,
    best_val: f64,
    best_gain: f64,
    left_indices: Vec<usize>,
    right_indices: Vec<usize>,
}

fn calc_gain_direct<'a>(
    train: &TrainDataSet,
    indices: &[usize],
    sum_grad: f64,
    sum_hessian: f64,
    params: &TreeParams,
    feature_id: usize,
) -> Option<SplitResult> {
    // sorted_instance_ids = instances[:, feature_id].argsort()
    let mut sorted_instance_ids = indices.to_vec();
    sorted_instance_ids.sort_unstable_by_key(|&row_id| train.features_rank[(row_id, feature_id)]);

    // Trivial cases: the feature is constant
    if sorted_instance_ids.is_empty() {
        return None;
    }
    let first = train.features[(*sorted_instance_ids.first().unwrap(), feature_id)];
    let last = train.features[(*sorted_instance_ids.last().unwrap(), feature_id)];
    if first == last {
        return None;
    }

    // We initialize at the first value
    let mut grad_left = train.grad[sorted_instance_ids[0]];
    let mut hessian_left = train.hessian[sorted_instance_ids[0]];
    let mut best_gain = -INFINITY;
    let mut best_idx = 0;
    let mut best_val = ::std::f64::NAN;
    let mut last_val = train.features[(sorted_instance_ids[0], feature_id)];

    // The potential split is before the current value, so we have to skip the first
    for (idx, &nrow) in sorted_instance_ids[..sorted_instance_ids.len()]
        .iter()
        .enumerate()
        .skip(1)
    {
        // We can only split when the value change
        let val = train.features[(nrow, feature_id)];
        if val != last_val {
            let current_gain = Node::_calc_split_gain(
                sum_grad,
                sum_hessian,
                grad_left,
                hessian_left,
                params.lambda,
                params.gamma,
            );

            if current_gain > best_gain {
                best_gain = current_gain;
                best_idx = idx;
                best_val = (val + last_val) / 2.;
            }
        }

        last_val = val;
        grad_left += train.grad[nrow];
        hessian_left += train.hessian[nrow];
    }

    // Sanity check
    assert!(!best_val.is_nan());

    let (left_indices, right_indices) = sorted_instance_ids.split_at(best_idx);
    Some(SplitResult {
        feature_id,
        best_val,
        best_gain,
        left_indices: left_indices.to_vec(),
        right_indices: right_indices.to_vec(),
    })
}

fn get_best_split_direct<'a>(
    train: &TrainDataSet,
    indices: &[usize],
    sum_grad: f64,
    sum_hessian: f64,
    params: &TreeParams,
) -> Option<SplitResult> {
    let results: Vec<_> = train
        .columns
        .iter()
        .filter_map(|&feature_id| {
            calc_gain_direct(&train, indices, sum_grad, sum_hessian, &params, feature_id)
        }).collect();
    results
        .into_iter()
        .ord_subset_max_by_key(|result| result.best_gain)
}

pub(crate) struct DirectReturn {
    pub(crate) node: Box<Node>,
    pub(crate) mean_val: f64,
}

/// Exact Greedy Algorithm for Split Findincg
///  (Refer to Algorithm1 of Reference[1])
pub(crate) fn build_direct(
    train: &TrainDataSet,
    indices: &[usize],
    predictions: &mut [f64],
    depth: usize,
    params: &TreeParams,
) -> DirectReturn {
    macro_rules! return_leaf {
        () => {{
            let mean_val =
                Node::_calc_leaf_weight(&train.grad, &train.hessian, params.lambda, indices);
            for &i in indices {
                predictions[i] = mean_val;
            }
            let node = Box::new(Node::Leaf(LeafNode { val: mean_val }));
            return DirectReturn { node, mean_val };
        }};
    }

    if depth >= params.max_depth {
        return_leaf!();
    }

    let sum_grad = sum_indices(&train.grad, indices);
    let sum_hessian = sum_indices(&train.hessian, indices);

    let best_result = get_best_split_direct(train, indices, sum_grad, sum_hessian, params);

    let best_result = match best_result {
        Some(e) => e,
        None => return_leaf!(),
    };

    if best_result.best_gain < params.min_split_gain {
        return_leaf!();
    }

    let left_child = Box::new(build_direct(
        &train,
        &best_result.left_indices,
        predictions,
        depth + 1,
        &params,
    ));

    let right_child = Box::new(build_direct(
        &train,
        &best_result.right_indices,
        predictions,
        depth + 1,
        &params,
    ));

    let mean_val = weighted_mean(
        right_child.mean_val,
        best_result.right_indices.len(),
        left_child.mean_val,
        best_result.left_indices.len(),
    );

    let node = Box::new(Node::Split(SplitNode {
        left_child: left_child.node,
        right_child: right_child.node,
        split_feature_id: best_result.feature_id,
        split_val: best_result.best_val,
        val: mean_val,
    }));

    DirectReturn { node, mean_val }
}
