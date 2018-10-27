use crate::{sum_indices, transmute_vec, LeafNode, Node, Params, SplitNode, TrainDataSet};
use ord_subset::OrdSubsetIterExt;
//use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use std::f64::INFINITY;
use std::mem::size_of;

/// Store the result of a successful split on a node
struct SplitResult {
    feature_id: usize,
    best_val: f64,
    best_gain: f64,
    left_indices: Vec<usize>,
    right_indices: Vec<usize>,
}

fn calc_gain_direct(
    train: &TrainDataSet,
    indices: &[usize],
    sum_grad: f64,
    sum_hessian: f64,
    params: &Params,
    feature_id: usize,
    cache: &mut [u8],
) -> Option<SplitResult> {
    // sorted_instance_ids = instances[:, feature_id].argsort()
    let sorted_instance_ids = &mut cache[0..indices.len() * size_of::<usize>()];
    let sorted_instance_ids: &mut [usize] = transmute_vec::<usize>(sorted_instance_ids);
    sorted_instance_ids.clone_from_slice(indices);
    sorted_instance_ids.sort_unstable_by_key(|&row_id| train.sorted_features[(row_id, feature_id)]);

    if sorted_instance_ids.first() == sorted_instance_ids.last() {
        return None;
    }

    // We initialize at the first value
    let mut grad_left = train.grad[sorted_instance_ids[0]];
    let mut hessian_left = train.hessian[sorted_instance_ids[0]];
    let mut best_gain = -INFINITY;
    let mut best_idx = 0;
    let mut best_val = train.features[(0, feature_id)];
    let mut last_val = train.features[(0, feature_id)];

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

    let (left_indices, right_indices) = sorted_instance_ids.split_at(best_idx);
    Some(SplitResult {
        feature_id,
        best_val,
        best_gain,
        left_indices: left_indices.to_vec(),
        right_indices: right_indices.to_vec(),
    })
}

fn get_best_split_direct(
    train: &TrainDataSet,
    indices: &[usize],
    sum_grad: f64,
    sum_hessian: f64,
    params: &Params,
    cache: &mut [u8],
) -> Option<SplitResult> {
    let cache: Vec<_> = cache
        .chunks_mut(train.target.len() * size_of::<usize>())
        .take(train.features.n_cols())
        .enumerate()
        .collect();
    let results: Vec<SplitResult> = cache
        .into_iter()
        .filter_map(|(feature_id, cache)| {
            calc_gain_direct(
                &train,
                indices,
                sum_grad,
                sum_hessian,
                &params,
                feature_id,
                cache,
            )
        }).collect();
    results
        .into_iter()
        .ord_subset_max_by_key(|result| result.best_gain)
}

/// Exact Greedy Algorithm for Split Finding
///  (Refer to Algorithm1 of Reference[1])
pub(crate) fn build_direct(
    train: &TrainDataSet,
    indices: &[usize],
    predictions: &mut [f64],
    depth: usize,
    params: &Params,
    cache: &mut [u8],
) -> Node {
    macro_rules! return_leaf {
        () => {{
            let val = Node::_calc_leaf_weight(&train.grad, &train.hessian, params.lambda, indices);
            for &i in indices {
                predictions[i] = val;
            }
            return Node::Leaf(LeafNode { val });
        }};
    }

    if depth >= params.max_depth {
        return_leaf!();
    }

    let sum_grad = sum_indices(&train.grad, indices);
    let sum_hessian = sum_indices(&train.hessian, indices);

    let best_result = get_best_split_direct(train, indices, sum_grad, sum_hessian, params, cache);

    let best_result: SplitResult = match best_result {
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
        cache,
    ));

    let right_child = Box::new(build_direct(
        &train,
        &best_result.right_indices,
        predictions,
        depth + 1,
        &params,
        cache,
    ));

    Node::Split(SplitNode {
        left_child,
        right_child,
        split_feature_id: best_result.feature_id,
        split_val: best_result.best_val,
    })
}

pub(crate) fn get_cache_size_direct(train: &TrainDataSet) -> usize {
    train.features.flat().len() * size_of::<usize>()
}
