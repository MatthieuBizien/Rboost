use crate::{ColumnMajorMatrix, Params, StridedVecView, TrainDataSet};
use ord_subset::OrdSubsetIterExt;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use std::f64::INFINITY;
use std::mem::size_of;

fn sum_indices(v: &[f64], indices: &[usize]) -> f64 {
    // A sum over a null set is not possible there, and this catch bugs.
    // The speed difference is negligible
    assert_ne!(indices.len(), 0);
    let mut o = 0.;
    for &i in indices {
        o += v[i];
    }
    o
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub(crate) struct SplitNode {
    left_child: Box<Node>,
    right_child: Box<Node>,
    split_feature_id: usize,
    split_val: f64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub(crate) struct LeafNode {
    val: f64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub(crate) enum Node {
    Split(SplitNode),
    Leaf(LeafNode),
}

/// Store the result of a successful split on a node
struct SplitResult {
    feature_id: usize,
    best_val: f64,
    best_gain: f64,
    left_indices: Vec<usize>,
    right_indices: Vec<usize>,
}

fn transmute_vec<T: Sized>(v: &mut [u8]) -> &mut [T] {
    //assert!(v.len() % size_of::<T>() == 0);
    unsafe { std::slice::from_raw_parts_mut(v.as_ptr() as *mut T, v.len() / size_of::<T>()) }
}

fn split_at_mut_transmute<T: Sized>(v: &mut [u8], n_elements: usize) -> (&mut [T], &mut [u8]) {
    let n_bytes = n_elements * size_of::<T>();
    //assert!(v.len() >= n_bytes);
    let (a, b) = v.split_at_mut(n_bytes);
    (transmute_vec(a), b)
}

impl Node {
    pub fn build_cache(train: &TrainDataSet, _params: &Params) -> Vec<u8> {
        (0..train.features.flat().len() * size_of::<usize>())
            .map(|_| 0)
            .collect()
    }

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
        cache: &mut [u8],
    ) -> Option<SplitResult> {
        // sorted_instance_ids = instances[:, feature_id].argsort()
        let sorted_instance_ids = &mut cache[0..indices.len() * size_of::<usize>()];
        let mut sorted_instance_ids: &mut [usize] = transmute_vec::<usize>(sorted_instance_ids);
        sorted_instance_ids.clone_from_slice(indices);
        sorted_instance_ids
            .sort_unstable_by_key(|&row_id| train.sorted_features[(row_id, feature_id)]);

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
                let current_gain = Self::_calc_split_gain(
                    sum_grad,
                    sum_hessian,
                    grad_left,
                    hessian_left,
                    param.lambda,
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
        param: &Params,
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
                Self::calc_gain_direct(
                    &train,
                    indices,
                    sum_grad,
                    sum_hessian,
                    &param,
                    feature_id,
                    cache,
                )
            }).collect();
        results
            .into_iter()
            .ord_subset_max_by_key(|result| result.best_gain)
    }

    fn calc_gain_bins(
        train: &TrainDataSet,
        indices: &[usize],
        sum_grad: f64,
        sum_hessian: f64,
        param: &Params,
        feature_id: usize,
        cache: &mut [u8],
    ) -> Option<(usize, f64, usize)> {
        let n_bin = train.n_bins[feature_id];

        let (grads, cache) = split_at_mut_transmute::<f64>(cache, n_bin);
        let (hessians, cache) = split_at_mut_transmute::<f64>(cache, n_bin);
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

        return Some((feature_id, best_gain, best_bin));
    }

    fn get_best_split_bins(
        train: &TrainDataSet,
        indices: &[usize],
        sum_grad: f64,
        sum_hessian: f64,
        param: &Params,
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
                Self::calc_gain_bins(
                    &train,
                    indices,
                    sum_grad,
                    sum_hessian,
                    &param,
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
        let mut best_val_left = -INFINITY;
        let mut best_val_right = INFINITY;
        for &i in indices {
            let bin = train.bins[(i, feature_id)] as usize;
            if bin <= best_bin {
                left_indices.push(i);
                best_val_left = best_val_left.max(train.features[(i, feature_id)]);
            } else {
                right_indices.push(i);
                best_val_right = best_val_right.min(train.features[(i, feature_id)]);
            }
        }
        let best_val = (best_val_left + best_val_right) / 2.;
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
    pub(crate) fn build(
        train: &TrainDataSet,
        indices: &[usize],
        predictions: &mut [f64],
        shrinkage_rate: f64,
        depth: usize,
        param: &Params,
        cache: &mut [u8],
    ) -> Node {
        macro_rules! return_leaf {
            () => {{
                let val =
                    Node::_calc_leaf_weight(&train.grad, &train.hessian, param.lambda, indices)
                        * shrinkage_rate;
                for &i in indices {
                    predictions[i] = val;
                }
                return Node::Leaf(LeafNode { val });
            }};
        }

        if depth >= param.max_depth {
            return_leaf!();
        }

        let sum_grad = sum_indices(&train.grad, indices);
        let sum_hessian = sum_indices(&train.hessian, indices);

        let best_result = if param.n_bins > 0 {
            let o = Self::get_best_split_bins(train, indices, sum_grad, sum_hessian, param, cache);
            o
        } else {
            let o =
                Self::get_best_split_direct(train, indices, sum_grad, sum_hessian, param, cache);
            o
        };

        let best_result: SplitResult = match best_result {
            Some(e) => e,
            None => return_leaf!(),
        };

        if best_result.best_gain < param.min_split_gain {
            return_leaf!();
        }

        let left_child = Box::new(Self::build(
            &train,
            &best_result.left_indices,
            predictions,
            shrinkage_rate,
            depth + 1,
            &param,
            cache,
        ));

        let right_child = Box::new(Self::build(
            &train,
            &best_result.right_indices,
            predictions,
            shrinkage_rate,
            depth + 1,
            &param,
            cache,
        ));

        Node::Split(SplitNode {
            left_child,
            right_child,
            split_feature_id: best_result.feature_id,
            split_val: best_result.best_val,
        })
    }

    pub fn predict(&self, features: &StridedVecView<f64>) -> f64 {
        match &self {
            Node::Split(split) => {
                let val = features[split.split_feature_id];
                if val <= split.split_val {
                    split.left_child.predict(&features)
                } else {
                    split.right_child.predict(&features)
                }
            }
            Node::Leaf(leaf) => leaf.val,
        }
    }

    pub fn par_predict(&self, features: &ColumnMajorMatrix<f64>) -> Vec<f64> {
        (0..features.n_rows())
            .into_iter()
            .map(|i| {
                let row = features.row(i);
                self.predict(&row)
            }).collect()
    }
}

#[cfg(test)]
mod tests {
    use crate::*;
    use failure::Error;

    fn parse_tsv(data: &str) -> Result<Dataset, Error> {
        let mut target: Vec<f64> = Vec::new();
        let mut features: Vec<Vec<f64>> = Vec::new();
        for l in data.split("\n") {
            if l.len() == 0 {
                continue;
            }
            let mut items = l.split("\t").into_iter();
            target.push(items.next().expect("first item").parse()?);
            features.push(items.map(|e| e.parse().unwrap()).collect());
        }
        let features = ColumnMajorMatrix::from_rows(features);

        Ok(Dataset { features, target })
    }

    #[test]
    fn test_regression() {
        let train = include_str!("../data/regression.train");
        let train = parse_tsv(train).expect("Train data");
        let test = include_str!("../data/regression.test");
        let test = parse_tsv(test).expect("Train data");

        let loss = RegLoss::default();
        let mut train = train.as_train_data(128);
        let zero_vec: Vec<_> = train.target.iter().map(|_| 0.).collect();
        train.update_grad_hessian(&loss, &zero_vec);

        let mut predictions: Vec<_> = train.target.iter().map(|_| 0.).collect();
        let indices: Vec<_> = (0..train.target.len()).collect();

        let params = Params {
            gamma: 0.,
            lambda: 1.,
            learning_rate: 0.99,
            max_depth: 6,
            min_split_gain: 0.1,
            n_bins: 10_000,
        };

        let mut cache: Vec<u8> = Node::build_cache(&train, &params);
        let tree = Node::build(
            &train,
            &indices,
            &mut predictions,
            1.,
            0,
            &params,
            &mut cache,
        );
        let pred2 = tree.par_predict(&train.features);
        assert_eq!(predictions.len(), pred2.len());
        for i in 0..predictions.len() {
            assert_eq!(predictions[i], pred2[i]);
        }

        let loss_train = rmse(&train.target, &predictions);
        let loss_test = rmse(&test.target, &tree.par_predict(&test.features));
        assert!(
            loss_train <= 0.433,
            "Train loss too important, expected 0.43062595, got {} (test {})",
            loss_train,
            loss_test
        );
        assert!(
            loss_test <= 0.446,
            "Test loss too important, expected 0.44403195, got {}",
            loss_test
        );
    }
}
