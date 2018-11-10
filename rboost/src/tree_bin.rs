use crate::tree_direct::build_direct;
use crate::{sum_indices, LeafNode, Node, SplitNode, TrainDataSet, TreeParams};
use ord_subset::OrdSubsetIterExt;
use std::f64::INFINITY;
use std::mem::size_of;

// Minimum number of rows before it's faster to use the direct algorithm
static MIN_ROWS_FOR_BINNING: usize = 100;

/// Store the result of a successful split on a node
struct SplitResult {
    feature_id: usize,
    best_val: f64,
    best_gain: f64,
    left_indices: Vec<usize>,
    right_indices: Vec<usize>,
}

fn update_grad_hessian(
    train: &TrainDataSet,
    indices: &[usize],
    feature_id: usize,
    grads: &mut [f64],
    hessians: &mut [f64],
) -> (usize, usize) {
    assert_eq!(grads.len(), hessians.len());
    for x in grads.iter_mut() {
        *x = 0.;
    }
    for x in hessians.iter_mut() {
        *x = 0.;
    }

    let mut min_bin = grads.len(); // placeholder value: if it don't change we have no data
    let mut max_bin = 0;

    for &i in indices {
        let bin = train.bins[(i, feature_id)] as usize;
        grads[bin] += train.grad[i];
        hessians[bin] += train.hessian[i];
        min_bin = min_bin.min(bin);
        max_bin = max_bin.max(bin);
    }
    (min_bin, max_bin)
}

fn calc_gain_bins(
    sum_grad: f64,
    sum_hessian: f64,
    params: &TreeParams,
    feature_id: usize,
    grads: &[f64],
    hessians: &[f64],
    min_bin: usize,
    max_bin: usize,
) -> Option<(usize, f64, usize)> {
    if max_bin == min_bin || max_bin == 0 {
        // Not possible to split if there is just one bin
        return None;
    }

    // We initialize at the first value
    let mut grad_left = 0.;
    let mut hessian_left = 0.;
    let mut best_gain = -INFINITY;
    let mut best_bin = 0;
    for bin in min_bin..max_bin {
        grad_left += grads[bin];
        hessian_left += hessians[bin];

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
            best_bin = bin;
        }
    }

    return Some((feature_id, best_gain, best_bin));
}

fn get_best_split_bins(
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
            let n_bins = train.n_bins[feature_id];
            let mut grads: Vec<_> = (0..n_bins).map(|_| 0.).collect();
            let mut hessians: Vec<_> = (0..n_bins).map(|_| 0.).collect();
            let (min_bin, max_bin) =
                update_grad_hessian(&train, &indices, feature_id, &mut grads, &mut hessians);
            calc_gain_bins(
                sum_grad,
                sum_hessian,
                &params,
                feature_id,
                &grads,
                &hessians,
                min_bin,
                max_bin,
            )
        }).collect();
    let best = results.into_iter().ord_subset_max_by_key(|result| result.1);
    let (feature_id, best_gain, best_bin) = match best {
        None => return None,
        Some(e) => e,
    };

    let mut left_indices = Vec::new();
    let mut right_indices = Vec::new();
    for &i in indices {
        let bin = train.bins[(i, feature_id)] as usize;
        if bin <= best_bin {
            left_indices.push(i);
        } else {
            right_indices.push(i);
        }
    }
    let best_val = train.threshold_vals[feature_id][best_bin];
    Some(SplitResult {
        feature_id,
        best_val,
        best_gain,
        left_indices,
        right_indices,
    })
}

/// Exact Greedy Algorithm for Split Finding
///  (Refer to Algorithm1 of Reference[1])
pub(crate) fn build_bins<'a>(
    train: &TrainDataSet,
    indices: &[usize],
    predictions: &mut [f64],
    depth: usize,
    params: &TreeParams,
) -> (Box<Node>, Option<&'a [f64]>) {
    // If the number of indices is too small it's faster to just use the direct algorithm
    if indices.len() <= MIN_ROWS_FOR_BINNING {
        let node = build_direct(train, indices, predictions, depth, params);
        return (Box::new(node), None);
    }
    macro_rules! return_leaf {
        () => {{
            let val = Node::_calc_leaf_weight(&train.grad, &train.hessian, params.lambda, indices);
            for &i in indices {
                predictions[i] = val;
            }
            return (Box::new(Node::Leaf(LeafNode { val })), None);
        }};
    }

    if depth >= params.max_depth {
        return_leaf!();
    }

    let sum_grad = sum_indices(&train.grad, indices);
    let sum_hessian = sum_indices(&train.hessian, indices);

    let best_result;
    {
        best_result = get_best_split_bins(train, indices, sum_grad, sum_hessian, params);
    }

    let best_result: SplitResult = match best_result {
        Some(e) => e,
        None => return_leaf!(),
    };

    if best_result.best_gain < params.min_split_gain {
        return_leaf!();
    }

    let left_child = build_bins(
        &train,
        &best_result.left_indices,
        predictions,
        depth + 1,
        &params,
    ).0;

    let right_child = build_bins(
        &train,
        &best_result.right_indices,
        predictions,
        depth + 1,
        &params,
    ).0;

    let node = Box::new(Node::Split(SplitNode {
        left_child,
        right_child,
        split_feature_id: best_result.feature_id,
        split_val: best_result.best_val,
    }));
    (node, None)
}
