use crate::{
    sum_indices, ColumnMajorMatrix, PreparedDataset, StridedVecView, TrainDataset, DEFAULT_GAMMA,
    DEFAULT_LAMBDA, DEFAULT_MAX_DEPTH, DEFAULT_MIN_SPLIT_GAIN,
};
//use rayon::prelude::ParallelIterator;
use crate::losses::Loss;
use crate::tree_bin::build_bins;
use crate::tree_direct::build_direct;

/// Parameters for building the tree.
///
/// They are the same than Xgboost https://xgboost.readthedocs.io/en/latest/parameter.html
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TreeParams {
    pub gamma: f64,
    pub lambda: f64,
    pub max_depth: usize,
    pub min_split_gain: f64,
}

impl TreeParams {
    pub fn new() -> Self {
        TreeParams {
            gamma: DEFAULT_GAMMA,
            lambda: DEFAULT_LAMBDA,
            max_depth: DEFAULT_MAX_DEPTH,
            min_split_gain: DEFAULT_MIN_SPLIT_GAIN,
        }
    }
}

impl Default for TreeParams {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub(crate) enum NanBranch {
    None,
    Left,
    Right,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub(crate) struct SplitNode {
    pub(crate) left_child: Box<Node>,
    pub(crate) right_child: Box<Node>,
    pub(crate) split_feature_id: usize,
    pub(crate) split_val: f64,
    pub(crate) val: f64,
    pub(crate) nan_branch: NanBranch,
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
        sum_indices(grad, indices) / (sum_indices(hessian, indices) + lambda)
    }

    pub(crate) fn build_from_train_data(
        train: &TrainDataset,
        indices: &[usize],
        predictions: &mut [f64],
        params: &TreeParams,
    ) -> Node {
        let depth = 0;
        let has_bins = train.n_bins.iter().any(|&n_bins| n_bins > 0);
        if has_bins {
            let out = build_bins(train, indices, predictions, depth, params);
            *(out.node)
        } else {
            let out = build_direct(train, indices, predictions, depth, params);
            *(out.node)
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
                if val.is_nan() {
                    match split.nan_branch {
                        NanBranch::Left => split.left_child.predict(&features),
                        NanBranch::Right => split.right_child.predict(&features),
                        // If we didn't see any NAN in the train branch
                        NanBranch::None => split.val,
                    }
                } else if val <= split.split_val {
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
            .map(|i| {
                let row = features.row(i);
                self.predict(&row)
            })
            .collect()
    }
}

/// Decision Tree implementation.
pub struct DecisionTree<L: Loss> {
    root: Node,
    loss: L,
}

impl<L: Loss> DecisionTree<L> {
    pub fn build(
        train: &PreparedDataset,
        predictions: &mut [f64],
        params: &TreeParams,
        loss: L,
    ) -> Self {
        let mut indices: Vec<_> = (0..train.target.len()).collect();
        let train = train.as_train_data(&loss);
        let node = Node::build_from_train_data(&train, &mut indices, predictions, params);
        DecisionTree { root: node, loss }
    }
    pub fn predict(&self, features: &StridedVecView<f64>) -> f64 {
        let o = self.root.predict(features);
        self.loss.get_target(o)
    }

    pub fn par_predict(&self, features: &ColumnMajorMatrix<f64>) -> Vec<f64> {
        self.root
            .par_predict(features)
            .iter()
            .map(|&o| self.loss.get_target(o))
            .collect()
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
        let train = train.as_prepared_data(3_000);

        let mut predictions: Vec<_> = train.target.iter().map(|_| 0.).collect();
        let mut params = TreeParams::new();
        params.max_depth = 6;

        let tree = DecisionTree::build(&train, &mut predictions, &params, loss);
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
        let train = train.as_prepared_data(0);

        let mut predictions: Vec<_> = train.target.iter().map(|_| 0.).collect();
        let mut params = TreeParams::new();
        params.max_depth = 6;

        let tree = DecisionTree::build(&train, &mut predictions, &params, loss);
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
