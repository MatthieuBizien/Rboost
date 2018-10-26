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

fn calc_gain_bins(
    train: &TrainDataSet,
    indices: &[usize],
    sum_grad: f64,
    sum_hessian: f64,
    params: &Params,
    feature_id: usize,
    cache: &mut [u8],
) -> Option<(usize, f64, usize)> {
    let n_bin = train.n_bins[feature_id];
    let (grads, cache) = split_at_mut_transmute::<f64>(cache, n_bin);
    let (hessians, _) = split_at_mut_transmute::<f64>(cache, n_bin);
    for x in grads.iter_mut() {
        *x = 0.;
    }
    for x in hessians.iter_mut() {
        *x = 0.;
    }

    let mut min_bin = n_bin; // placeholder value: if it don't change we have no data
    let mut max_bin = 0;

    for &i in indices {
        let bin = train.bins[(i, feature_id)] as usize;
        grads[bin] += train.grad[i];
        hessians[bin] += train.hessian[i];
        min_bin = min_bin.min(bin);
        max_bin = max_bin.max(bin);
    }
    if max_bin == min_bin || min_bin == n_bin {
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

fn get_best_split_bins(
    train: &TrainDataSet,
    indices: &[usize],
    sum_grad: f64,
    sum_hessian: f64,
    params: &Params,
    cache: &mut [u8],
) -> Option<SplitResult> {
    let size_element = 2 * size_of::<f64>();
    let mut caches: Vec<_> = Vec::new();
    let mut cache = &mut cache[..];
    for &size_bin in &train.n_bins {
        let e = cache.split_at_mut(size_bin * size_element);
        cache = e.1;
        caches.push(e.0);
    }
    let caches: Vec<_> = caches.into_iter().enumerate().collect();
    let results: Vec<_> = caches
        .into_iter()
        .filter_map(|(feature_id, cache)| {
            calc_gain_bins(
                &train,
                indices,
                sum_grad,
                sum_hessian,
                &params,
                feature_id,
                cache,
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
pub(crate) fn build_bins(
    train: &TrainDataSet,
    indices: &[usize],
    predictions: &mut [f64],
    shrinkage_rate: f64,
    depth: usize,
    params: &Params,
    cache: &mut [u8],
) -> Node {
    macro_rules! return_leaf {
        () => {{
            let val = Node::_calc_leaf_weight(&train.grad, &train.hessian, params.lambda, indices)
                * shrinkage_rate;
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

    let best_result = get_best_split_bins(train, indices, sum_grad, sum_hessian, params, cache);

    let best_result: SplitResult = match best_result {
        Some(e) => e,
        None => return_leaf!(),
    };

    if best_result.best_gain < params.min_split_gain {
        return_leaf!();
    }

    let left_child = Box::new(build_bins(
        &train,
        &best_result.left_indices,
        predictions,
        shrinkage_rate,
        depth + 1,
        &params,
        cache,
    ));

    let right_child = Box::new(build_bins(
        &train,
        &best_result.right_indices,
        predictions,
        shrinkage_rate,
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

pub(crate) fn get_cache_size_bin(train: &TrainDataSet) -> usize {
    train.n_bins.iter().sum::<usize>() * size_of::<f64>() * 2
}
