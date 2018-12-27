use crate::{
    weighted_mean, ColumnMajorMatrix, LeafNode, NanBranch, Node, SplitNode, TrainDataset,
    TreeParams, SHOULD_NOT_HAPPEN,
};
use either::Either;
use rayon::prelude::*;
use std::f64::INFINITY;

#[derive(Debug)]
pub(crate) struct Bin2Return {
    pub(crate) node: Node,
    pub(crate) mean_val: f64,
    pub(crate) n_obs: usize,
}

/// Just a range function that can works in reverse:
/// range(0, 5) = [0, 1, 2, 3, 4]
/// range(5, 0) = [5, 4, 3, 2, 1]
fn range(start: usize, end: usize) -> Box<Iterator<Item = usize>> {
    if start < end {
        Box::new(start..end)
    } else {
        Box::new((end..start).map(move |e| start - e)) // TODO check -1
    }
}

/// Compute the best gain for every possible split between bins
fn compute_gain(
    grads: &[f64],
    hessians: &[f64],
    is_empty: &[bool],
    start: usize,
    end: usize,
    sum_grad: f64,
    sum_hessian: f64,
    params: &TreeParams,
) -> (f64, usize) {
    // We initialize at the first value
    let mut grad_left = 0.;
    let mut hessian_left = 0.;
    let mut best_gain = -INFINITY;
    let mut best_bin = 0;
    for bin in range(start, end) {
        if is_empty[bin] {
            // continue;
        }
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
}

/// Get the index of the node if the node is between min_node and max_node, or None otherwise
fn get_node_idx(node: usize, min_node: usize, max_node: usize) -> Option<usize> {
    if (node >= min_node) & (node < max_node) {
        Some(node - min_node)
    } else {
        None
    }
}

#[derive(Clone, Debug)]
struct GainResult {
    feature_id: usize,
    gain: f64,
    bin: usize,
    nan_branch: NanBranch,
}

/// Compute the gain if we split for this feature on all the nodes.
fn calc_gains_bin(
    train: &TrainDataset,
    nodes_of_rows: &[usize],
    sums_grad: &[f64],
    sums_hessian: &[f64],
    params: &TreeParams,
    feature_id: usize,
    (min_node, max_node): (usize, usize),
) -> Vec<Option<GainResult>> {
    let n_bins = train.n_bins[feature_id];
    let n_nodes = max_node - min_node;

    let mut grads = ColumnMajorMatrix::from_function(n_bins, n_nodes, |_, _| 0.);
    let mut hessians = ColumnMajorMatrix::from_function(n_bins, n_nodes, |_, _| 0.);
    let mut is_empty = ColumnMajorMatrix::from_function(n_bins, n_nodes, |_, _| true);

    // placeholder value: if it don't change we have no data
    let mut min_bin = vec![n_bins; n_nodes];
    let mut max_bin = vec![0; n_nodes];
    let mut n_nan = vec![0; n_nodes];

    // We iterate over all the data to compute the values of the grads and hessians per bin and node
    for i in 0..train.n_rows() {
        if let Some(node) = get_node_idx(nodes_of_rows[i], min_node, max_node) {
            match train.bins[(i, feature_id)] {
                Some(bin) => {
                    let bin = bin as usize;
                    min_bin[node] = min_bin[node].min(bin);
                    max_bin[node] = max_bin[node].max(bin);
                    grads[(bin, node)] += train.grad[i];
                    hessians[(bin, node)] += train.hessian[i];
                    is_empty[(bin, node)] = false;
                }
                None => {
                    // NAN values are implicitly in sum_grad and sum_hessian
                    n_nan[node] += 1;
                }
            }
        }
    }

    (0..n_nodes)
        .map(|node| {
            // Helper function
            let compute_gain = |start, end| {
                compute_gain(
                    grads.column(node),
                    hessians.column(node),
                    is_empty.column(node),
                    start,
                    end,
                    sums_grad[node],
                    sums_hessian[node],
                    params,
                )
            };

            // First pass: we loop over all the bins left to right.
            let (best_gain, best_bin) = compute_gain(min_bin[node], max_bin[node]);
            if n_nan[node] == 0 {
                // Short path if there is no NAN
                return Some(GainResult {
                    feature_id,
                    gain: best_gain,
                    bin: best_bin,
                    nan_branch: NanBranch::None,
                });
            }

            // If there is NAN, we try to get the best path in the reverse order, so we can choose if the
            // default path for NAN should be in the right branch or the left branch.
            let (best_gain_rev, best_bin_rev) = compute_gain(max_bin[node], min_bin[node]);

            if best_gain > best_gain_rev {
                // For the "left to right" order, the NAN are implicitly in the left branch.
                Some(GainResult {
                    feature_id,
                    gain: best_gain,
                    bin: best_bin,
                    nan_branch: NanBranch::Right,
                })
            } else {
                // It's the opposite for the "right to left" order
                Some(GainResult {
                    feature_id,
                    gain: best_gain_rev,
                    bin: best_bin_rev,
                    nan_branch: NanBranch::Left,
                })
            }
        })
        .collect()
}

fn get_best_splits_bin(
    train: &TrainDataset,
    nodes_of_rows: &[usize],
    sums_grad: &[f64],
    sums_hessian: &[f64],
    params: &TreeParams,
    (min_node, max_node): (usize, usize),
) -> Box<Iterator<Item = Option<GainResult>>> {
    // Per node and per feature, what is the best split
    let best_split_per_node_and_feature: Vec<Vec<_>> = train
        .columns
        .par_iter()
        .map(|&feature_id| {
            calc_gains_bin(
                &train,
                nodes_of_rows,
                sums_grad,
                sums_hessian,
                &params,
                feature_id,
                (min_node, max_node),
            )
        })
        .collect();

    // We concatenate: per node, what is the best split
    Box::new((0..(max_node - min_node)).map(move |node| {
        let mut best_result: Option<GainResult> = None;
        for col in &best_split_per_node_and_feature {
            if let Some(result) = &col[node] {
                if let Some(best_) = &best_result {
                    if result.gain >= best_.gain {
                        best_result = Some(result.clone());
                    }
                } else {
                    // Default behaviour for the first iteration
                    best_result = Some(result.clone());
                }
            }
        }
        best_result
    }))
}

/// Exact Greedy Algorithm for Split Finding
pub(crate) fn build_bins2(
    train: &TrainDataset,
    nodes_of_rows: &mut [usize],
    predictions: &mut [f64],
    depth: usize,
    params: &TreeParams,
    min_node: usize,
    max_node: usize,
) -> Vec<Bin2Return> {
    debug_assert!(max_node > min_node);

    let mut sums_grad = vec![0.; max_node - min_node];
    let mut sums_hessian = vec![0.; max_node - min_node];
    let mut n_obs_per_node = vec![0; max_node - min_node];
    for i in 0..train.n_rows() {
        if let Some(node) = get_node_idx(nodes_of_rows[i], min_node, max_node) {
            sums_grad[node] += train.grad[i];
            sums_hessian[node] += train.hessian[i];
            n_obs_per_node[node] += 1;
        }
    }

    let best_splits = if depth >= params.max_depth {
        // Fast path if nothing to do
        Box::new((min_node..max_node).map(|_| None))
    } else {
        // Per node, what is the best split
        get_best_splits_bin(
            train,
            nodes_of_rows,
            &sums_grad,
            &sums_hessian,
            params,
            (min_node, max_node),
        )
    };
    // Add an index to every node that we will split.
    // We pre-compute for the leaf the final value
    let mut n_with_child = 0;
    let nodes: Vec<_> = best_splits
        .zip(sums_grad)
        .zip(sums_hessian)
        .map(|((split, sum_grad), sum_hessian)| {
            if let Some(split) = split {
                if split.gain >= params.min_split_gain {
                    let o = Either::Left((n_with_child, split));
                    n_with_child += 1;
                    return o;
                }
            }
            return Either::Right(sum_grad / (sum_hessian + params.lambda));
        })
        .collect();

    // Update the predictions for nodes that will not be split
    if n_with_child != nodes.len() {
        for i in 0..train.n_rows() {
            if let Some(node) = get_node_idx(nodes_of_rows[i], min_node, max_node) {
                if let Either::Right(val) = nodes[node] {
                    predictions[i] = val;
                }
            }
        }
    }

    // Fast path if nothing can be split
    if n_with_child == 0 {
        return n_obs_per_node
            .into_iter()
            .zip(nodes)
            .map(|(n_obs, node)| {
                let mean_val = node.right().expect(SHOULD_NOT_HAPPEN);
                let node = LeafNode {
                    val: mean_val,
                    n_obs,
                };
                Bin2Return {
                    node: Node::Leaf(node),
                    mean_val,
                    n_obs,
                }
            })
            .collect();
    }

    // For every node, put them either on the left or on the right.
    // The indices of the new nodes starts at max_node and goes up 2 by 2 when we have a split.
    for i in 0..train.n_rows() {
        if let Some(node_idx) = get_node_idx(nodes_of_rows[i], min_node, max_node) {
            //let node_idx = nodes_of_rows[i] - min_node;
            if let Either::Left((n_node_with_child, split)) = &nodes[node_idx] {
                if let Some(bin) = train.bins[(i, split.feature_id)] {
                    let n_node_with_child = *n_node_with_child;
                    if bin as usize <= split.bin {
                        // Left child
                        nodes_of_rows[i] = n_node_with_child * 2 + max_node
                    } else {
                        // Right child
                        nodes_of_rows[i] = n_node_with_child * 2 + max_node + 1
                    }
                }
            }
        }
    }

    // We iterate again to compute the children
    let children = build_bins2(
        train,
        nodes_of_rows,
        predictions,
        depth + 1,
        params,
        max_node,
        max_node + n_with_child * 2,
    );
    debug_assert_eq!(children.len(), n_with_child * 2);
    let mut children = children.into_iter();

    // We add back the children to the parents and return
    let o = nodes
        .into_iter()
        .zip(n_obs_per_node)
        .map(|(e, n_obs)| match e {
            Either::Left((_, split)) => {
                let left_child = children.next().unwrap();
                debug_assert!(left_child.n_obs > 0);

                let right_child = children.next().unwrap();
                debug_assert!(right_child.n_obs > 0);

                debug_assert_eq!(left_child.n_obs + right_child.n_obs, n_obs);
                let mean_val = weighted_mean(
                    right_child.mean_val,
                    right_child.n_obs,
                    left_child.mean_val,
                    left_child.n_obs,
                );

                let node = SplitNode {
                    left_child: Box::new(left_child.node),
                    right_child: Box::new(right_child.node),
                    split_feature_id: split.feature_id,
                    split_val: train.threshold_vals[split.feature_id][split.bin],
                    val: mean_val,
                    nan_branch: split.nan_branch,
                    n_obs,
                };
                Bin2Return {
                    node: Node::Split(node),
                    mean_val,
                    n_obs,
                }
            }
            Either::Right(mean_val) => {
                let node = LeafNode {
                    val: mean_val,
                    n_obs,
                };
                Bin2Return {
                    node: Node::Leaf(node),
                    mean_val,
                    n_obs,
                }
            }
        })
        .collect();

    debug_assert!(children.next().is_none());
    o
}
