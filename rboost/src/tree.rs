use crate::{sum_indices, ColumnMajorMatrix, Params, StridedVecView, TrainDataSet};
//use rayon::prelude::ParallelIterator;
use crate::tree_bin::{build_bins, get_cache_size_bin};
use crate::tree_direct::{build_direct, get_cache_size_direct};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub(crate) struct SplitNode {
    pub(crate) left_child: Box<Node>,
    pub(crate) right_child: Box<Node>,
    pub(crate) split_feature_id: usize,
    pub(crate) split_val: f64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub(crate) struct LeafNode {
    pub(crate) val: f64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub(crate) enum Node {
    Split(SplitNode),
    Leaf(LeafNode),
}

impl Node {
    ///  Loss reduction
    /// (Refer to Eq7 of Reference[1])
    pub(crate) fn _calc_split_gain(
        g: f64,
        h: f64,
        g_l: f64,
        h_l: f64,
        lambda: f64,
        gamma: f64,
    ) -> f64 {
        fn calc_term(g: f64, h: f64, lambda: f64) -> f64 {
            g.powi(2) / (h + lambda)
        }
        let g_r = g - g_l;
        let h_r = h - h_l;
        calc_term(g_l, h_l, lambda) + calc_term(g_r, h_r, lambda) - calc_term(g, h, lambda) - gamma
    }

    /// Calculate the optimal weight of this leaf node.
    /// (Refer to Eq5 of Reference[1])
    pub(crate) fn _calc_leaf_weight(
        grad: &[f64],
        hessian: &[f64],
        lambda: f64,
        indices: &[usize],
    ) -> f64 {
        return sum_indices(grad, indices) / (sum_indices(hessian, indices) + lambda);
    }

    pub fn build(
        train: &TrainDataSet,
        indices: &[usize],
        predictions: &mut [f64],
        params: &Params,
        cache: &mut Vec<u8>,
    ) -> Node {
        let depth = 0;
        if params.n_bins > 0 {
            cache.resize(get_cache_size_bin(&train, &params), 0);
            let (boxed_node, _) =
                build_bins(train, indices, predictions, depth, params, cache, None);
            *boxed_node
        } else {
            cache.resize(get_cache_size_direct(&train), 0);
            build_direct(train, indices, predictions, depth, params, cache)
        }
    }

    pub fn apply_shrinking(&mut self, shrinkage_rate: f64) {
        match self {
            Node::Leaf(ref mut node) => node.val *= shrinkage_rate,
            Node::Split(ref mut node) => {
                Node::apply_shrinking(&mut node.left_child, shrinkage_rate);
                Node::apply_shrinking(&mut node.right_child, shrinkage_rate);
            }
        }
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

    #[test]
    fn test_regression_bins() {
        let train = include_str!("../data/regression.train");
        let train = parse_csv(train, "\t").expect("Train data");
        let test = include_str!("../data/regression.test");
        let test = parse_csv(test, "\t").expect("Train data");

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
            booster: Booster::Geometric,
        };

        let mut cache: Vec<u8> = Vec::new();
        let tree = Node::build(&train, &indices, &mut predictions, &params, &mut cache);
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

    #[test]
    fn test_regression_direct() {
        let train = include_str!("../data/regression.train");
        let train = parse_csv(train, "\t").expect("Train data");
        let test = include_str!("../data/regression.test");
        let test = parse_csv(test, "\t").expect("Train data");

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
            n_bins: 0,
            booster: Booster::Geometric,
        };

        let mut cache: Vec<u8> = Vec::new();
        let tree = Node::build(&train, &indices, &mut predictions, &params, &mut cache);
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
