use crate::tree_direct::build_direct;
use crate::{
    sum_indices, weighted_mean, LeafNode, NanBranch, Node, SplitNode, TrainDataset, TreeParams,
};
use ord_subset::OrdSubsetIterExt;
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

/// Just a range function that can works in reverse:
/// range(0, 5) = [0, 1, 2, 3, 4]
/// range(5, 0) = [5, 4, 3, 2, 1]
fn range(start: usize, end: usize) -> Box<Iterator<Item = usize>> {
    if start < end {
        Box::new(start..end)
    } else {
        Box::new((end..start).map(move |e| start - e))
    }
}

fn calc_gain_bins(
    train: &TrainDataset,
    indices: &[usize],
    feature_id: usize,
    sum_grad: f64,
    sum_hessian: f64,
    params: &TreeParams,
) -> Option<(usize, f64, usize, NanBranch)> {
    let n_bins = train.n_bins[feature_id];

    // First we compute the values of the bins on the dataset
    let mut grads: Vec<_> = (0..n_bins).map(|_| 0.).collect();
    let mut hessians: Vec<_> = (0..n_bins).map(|_| 0.).collect();
    let mut min_bin = grads.len(); // placeholder value: if it don't change we have no data
    let mut max_bin = 0;
    let mut n_nan = 0;

    // We iterate over all the indices. We currently don't have a fast path for sparse values.
    for &i in indices {
        match train.bins[(i, feature_id)] {
            Some(bin) => {
                let bin = bin as usize;
                min_bin = min_bin.min(bin);
                max_bin = max_bin.max(bin);
                grads[bin] += train.grad[i];
                hessians[bin] += train.hessian[i];
            }
            None => {
                // NAN values are implicitly in sum_grad and sum_hessian
                n_nan += 1;
            }
        };
    }

    if max_bin == min_bin || max_bin == 0 {
        // Not possible to split if there is just one bin
        return None;
    }

    // Compute the gain by looping over
    let compute_gain = |start, end| {
        // We initialize at the first value
        let mut grad_left = 0.;
        let mut hessian_left = 0.;
        let mut best_gain = -INFINITY;
        let mut best_bin = 0;
        for bin in range(start, end) {
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
        (best_gain, best_bin)
    };

    // First pass: we loop over all the bins left to right.
    let (best_gain, best_bin) = compute_gain(min_bin, max_bin);
    if n_nan == 0 {
        // Short path if there is no NAN
        return Some((feature_id, best_gain, best_bin, NanBranch::None));
    }

    // If there is NAN, we try to get the best path in the reverse order, so we can choose if the
    // default path for NAN should be in the right branch or the left branch.
    let (best_gain_rev, best_bin_rev) = compute_gain(max_bin, min_bin);

    if best_gain > best_gain_rev {
        // For the "left to right" order, the NAN are implicitly in the left branch.
        Some((feature_id, best_gain, best_bin, NanBranch::Right))
    } else {
        // It's the opposite for the "right to left" order
        Some((feature_id, best_gain_rev, best_bin_rev, NanBranch::Left))
    }
}

fn get_best_split_bins(
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
            calc_gain_bins(&train, &indices, feature_id, sum_grad, sum_hessian, &params)
        })
        .collect();
    let best = results.into_iter().ord_subset_max_by_key(|result| result.1);
    let (feature_id, best_gain, best_bin, nan_branch) = match best {
        None => return None,
        Some(e) => e,
    };

    let mut left_indices = Vec::new();
    let mut right_indices = Vec::new();
    for &i in indices {
        match train.bins[(i, feature_id)] {
            Some(bin) => {
                let bin = bin as usize;
                if bin <= best_bin {
                    left_indices.push(i);
                } else {
                    right_indices.push(i);
                }
            }
            None => {
                match nan_branch {
                    NanBranch::Left => left_indices.push(i),
                    NanBranch::Right => right_indices.push(i),
                    NanBranch::None => {} // We drop the indices if there is no preferred branch
                }
            }
        };
    }
    let best_val = train.threshold_vals[feature_id][best_bin];
    Some(SplitResult {
        feature_id,
        best_val,
        best_gain,
        left_indices,
        right_indices,
        nan_branch,
    })
}

pub(crate) struct SplitBinReturn {
    pub(crate) node: Box<Node>,
    pub(crate) mean_val: f64,
}

/// Exact Greedy Algorithm for Split Finding
///  (Refer to Algorithm1 of Reference[1])
pub(crate) fn build_bins(
    train: &TrainDataset,
    indices: &[usize],
    predictions: &mut [f64],
    depth: usize,
    params: &TreeParams,
) -> SplitBinReturn {
    // If the number of indices is too small it's faster to just use the direct algorithm
    if indices.len() <= params.min_rows_for_binning {
        let out = build_direct(train, indices, predictions, depth, params);
        return SplitBinReturn {
            node: out.node,
            mean_val: out.mean_val,
        };
    }

    macro_rules! return_leaf {
        () => {{
            let mean_val =
                Node::_calc_leaf_weight(&train.grad, &train.hessian, params.lambda, indices);
            for &i in indices {
                predictions[i] = mean_val;
            }
            let node = Box::new(Node::Leaf(LeafNode {
                val: mean_val,
                n_obs: indices.len(),
            }));
            return SplitBinReturn { node, mean_val };
        }};
    }

    if depth >= params.max_depth {
        return_leaf!();
    }

    let sum_grad = sum_indices(&train.grad, indices);
    let sum_hessian = sum_indices(&train.hessian, indices);

    let best_result = get_best_split_bins(train, indices, sum_grad, sum_hessian, params);

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
    );

    let right_child = build_bins(
        &train,
        &best_result.right_indices,
        predictions,
        depth + 1,
        &params,
    );

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
        nan_branch: best_result.nan_branch,
        n_obs: indices.len(),
    }));

    SplitBinReturn { node, mean_val }
}
