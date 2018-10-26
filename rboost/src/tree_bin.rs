use crate::{split_at_mut_transmute, sum_indices, LeafNode, Node, Params, SplitNode, TrainDataSet};
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
    params: &Params,
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
        );

        if current_gain > best_gain {
            best_gain = current_gain;
            best_bin = bin;
        }
    }

    return Some((feature_id, best_gain, best_bin));
}

fn split_grads_hessians<'a>(
    n_bins: &[usize],
    cache: &'a mut [f64],
) -> Vec<(usize, &'a mut [f64], &'a mut [f64])> {
    let mut caches: Vec<_> = Vec::new();
    let mut cache = &mut cache[..];
    for (i, &size_bin) in n_bins.iter().enumerate() {
        let (grads, cache_) = cache.split_at_mut(size_bin);
        let (hessians, cache_) = cache_.split_at_mut(size_bin);
        cache = cache_;
        caches.push((i, grads, hessians));
    }
    caches
}

fn get_best_split_bins(
    train: &TrainDataSet,
    indices: &[usize],
    sum_grad: f64,
    sum_hessian: f64,
    params: &Params,
    grads_hessians: &mut [f64],
) -> Option<SplitResult> {
    let caches: Vec<_> = split_grads_hessians(&train.n_bins, grads_hessians);
    let results: Vec<_> = caches
        .into_iter()
        .filter_map(|(feature_id, grads, hessians)| {
            let (min_bin, max_bin) =
                update_grad_hessian(&train, &indices, feature_id, grads, hessians);
            calc_gain_bins(
                sum_grad,
                sum_hessian,
                &params,
                feature_id,
                grads,
                hessians,
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
        left_indices: left_indices.to_vec(),
        right_indices: right_indices.to_vec(),
    })
}

/// Exact Greedy Algorithm for Split Finding
///  (Refer to Algorithm1 of Reference[1])
pub(crate) fn build_bins<'a>(
    train: &TrainDataSet,
    indices: &[usize],
    predictions: &mut [f64],
    shrinkage_rate: f64,
    depth: usize,
    params: &Params,
    cache: &'a mut [u8],
    grads_hessians: Option<&mut [f64]>,
) -> (Box<Node>, Option<&'a [f64]>) {
    macro_rules! return_leaf {
        () => {{
            let val = Node::_calc_leaf_weight(&train.grad, &train.hessian, params.lambda, indices)
                * shrinkage_rate;
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

    let grads_hessians = grads_hessians.unwrap_or_else(||{
        let n_elements = n_elements_per_node(&train);
        split_at_mut_transmute::<f64>(cache, n_elements).0
    });
    let best_result = get_best_split_bins(
        train,
        indices,
        sum_grad,
        sum_hessian,
        params,
        grads_hessians,
    );

    let best_result: SplitResult = match best_result {
        Some(e) => e,
        None => return_leaf!(),
    };

    if best_result.best_gain < params.min_split_gain {
        return_leaf!();
    }

    let (left_child, _) = build_bins(
        &train,
        &best_result.left_indices,
        predictions,
        shrinkage_rate,
        depth + 1,
        &params,
        cache,
        None,
    );

    let (right_child, _) = build_bins(
        &train,
        &best_result.right_indices,
        predictions,
        shrinkage_rate,
        depth + 1,
        &params,
        cache,
        None,
    );

    let node = Box::new(Node::Split(SplitNode {
        left_child,
        right_child,
        split_feature_id: best_result.feature_id,
        split_val: best_result.best_val,
    }));
    (node, None)
}

pub(crate) fn n_elements_per_node(train: &TrainDataSet) -> usize {
    train.n_bins.iter().sum::<usize>() * size_of::<f64>() * 2
}

pub(crate) fn get_cache_size_bin(train: &TrainDataSet) -> usize {
    n_elements_per_node(train) * size_of::<f64>()
}
