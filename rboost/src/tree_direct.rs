use crate::{
    sum_indices, weighted_mean, LeafNode, NanBranch, Node, SplitNode, TrainDataset, TreeParams,
};
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
    nan_branch: NanBranch,
}

// Because we use strict float ranking, we have to use exact float comparison.
#[allow(clippy::float_cmp)]
fn calc_gain_direct(
    train: &TrainDataset,
    indices: &[usize],
    sum_grad: f64,
    sum_hessian: f64,
    params: &TreeParams,
    feature_id: usize,
) -> Option<SplitResult> {
    // sorted_instance_ids = instances[:, feature_id].argsort()
    let mut sorted_instance_ids = indices.to_vec();
    sorted_instance_ids.sort_unstable_by_key(|&row_id| train.features_rank[(row_id, feature_id)]);

    for &idx in &sorted_instance_ids {
        assert!(!train.features[(idx, feature_id)].is_nan());
        assert_ne!(train.features_rank[(idx, feature_id)], 0);
    }

    // Number of NAN. We use the fact that the rank is 0 iif the value is NAN.
    let n_nan = sorted_instance_ids
        .iter()
        .enumerate()
        .filter(|(_, &idx)| train.features_rank[(idx, feature_id)] != 0)
        .map(|(n, _)| n)
        .next();
    let n_nan = match n_nan {
        None => return None, // We only have NAN values
        Some(e) => e,
    };

    // We check a trivial cases: the feature is constant.
    // Because the rank of the NAN is always smaller than the rank of the other values, at n_nan
    // we have the smallest not-nan value
    let first_rank = train.features_rank[(sorted_instance_ids[n_nan], feature_id)];
    let last_rank = train.features_rank[(*sorted_instance_ids.last().unwrap(), feature_id)];
    if first_rank == last_rank {
        return None;
    }

    let grad_nan: f64 = (0..n_nan).map(|n| train.grad[sorted_instance_ids[n]]).sum();
    let hessian_nan: f64 = (0..n_nan)
        .map(|n| train.hessian[sorted_instance_ids[n]])
        .sum();

    // We initialize at the first value
    let mut grad_left = train.grad[sorted_instance_ids[n_nan]];
    let mut hessian_left = train.hessian[sorted_instance_ids[n_nan]];
    let mut best_gain = -INFINITY;
    let mut best_idx = 0;
    let mut best_val = ::std::f64::NAN;
    let mut last_val = train.features[(sorted_instance_ids[n_nan], feature_id)];
    let mut best_nan_branch = NanBranch::None;

    // The potential split is before the current value, so we have to skip the first
    for (idx, &nrow) in sorted_instance_ids[n_nan..sorted_instance_ids.len()]
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
                best_nan_branch = if n_nan > 0 {
                    // If there is NAN they are implicitly in the right branch
                    NanBranch::Right
                } else {
                    // If there is no NAN we don't know what to do
                    NanBranch::None
                };
            }

            if n_nan > 0 {
                // We evaluate what happens if we push all the NAN values to the left branch
                let current_gain = Node::_calc_split_gain(
                    sum_grad,
                    sum_hessian,
                    grad_left + grad_nan,
                    hessian_left + hessian_nan,
                    params.lambda,
                    params.gamma,
                );

                if current_gain > best_gain {
                    best_gain = current_gain;
                    best_idx = idx;
                    best_val = (val + last_val) / 2.;
                    best_nan_branch = NanBranch::Left;
                }
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
        nan_branch: best_nan_branch,
    })
}

fn get_best_split_direct(
    train: &TrainDataset,
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
        })
        .collect();
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
    train: &TrainDataset,
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
        //nan_branch: NanBranch::None,
        nan_branch: best_result.nan_branch,
    }));

    DirectReturn { node, mean_val }
}
