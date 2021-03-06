use crate::{
    sum_indices, ColumnMajorMatrix, FitResult, PreparedDataset, StridedVecView, TrainDataset,
    DEFAULT_GAMMA, DEFAULT_LAMBDA, DEFAULT_MAX_DEPTH, DEFAULT_MIN_SPLIT_GAIN,
};
//use rayon::prelude::ParallelIterator;
use crate::losses::Loss;
use crate::tree_all::build_bins2;
use crate::tree_direct::build_direct;

/// Parameters for building the tree.
///
/// They are the same than Xgboost <https://xgboost.readthedocs.io/en/latest/parameter.html>
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct TreeParams {
    pub gamma: f64,
    pub lambda: f64,
    pub max_depth: usize,
    pub min_split_gain: f64,
    pub recursive_split: bool,
    pub min_rows_for_binning: usize,
}

impl TreeParams {
    pub fn new() -> Self {
        TreeParams {
            gamma: DEFAULT_GAMMA,
            lambda: DEFAULT_LAMBDA,
            max_depth: DEFAULT_MAX_DEPTH,
            min_split_gain: DEFAULT_MIN_SPLIT_GAIN,
            recursive_split: false,
            min_rows_for_binning: 0,
        }
    }
}

impl Default for TreeParams {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub(crate) enum NanBranch {
    None,
    Left,
    Right,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub(crate) struct SplitNode {
    pub(crate) split_feature_id: usize,
    pub(crate) n_obs: usize,
    pub(crate) split_val: f64,
    pub(crate) val: f64,
    pub(crate) nan_branch: NanBranch,
    pub(crate) left_child: Box<Node>,
    pub(crate) right_child: Box<Node>,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub(crate) struct LeafNode {
    pub(crate) val: f64,
    pub(crate) n_obs: usize,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
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
        assert_eq!(
            train
                .grad
                .iter()
                .map(|f| f.is_nan() as usize)
                .sum::<usize>(),
            0
        );
        assert_eq!(
            train
                .hessian
                .iter()
                .map(|f| f.is_nan() as usize)
                .sum::<usize>(),
            0
        );

        let depth = 0;
        let has_bins = train.n_bins.iter().any(|&n_bins| n_bins > 0);
        if has_bins {
            if params.recursive_split {
                return *crate::tree_bin::build_bins(train, indices, predictions, depth, params)
                    .node;
            }

            let mut nodes_of_rows = vec![::std::usize::MAX; train.n_rows()];
            for &i in indices {
                nodes_of_rows[i] = 0;
            }
            let out = build_bins2(train, &mut nodes_of_rows, predictions, depth, params, 0, 1);
            assert_eq!(out.len(), 1);
            let out = match out.into_iter().next() {
                Some(e) => e.node,
                _ => panic!(),
            };

            out
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
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
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
    ) -> FitResult<Self> {
        let indices: Vec<_> = (0..train.target.len()).collect();
        let train = train.as_train_data(&loss);
        let node = Node::build_from_train_data(&train, &indices, predictions, params);
        Ok(Self { root: node, loss })
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
    fn test_regression_bins() -> Result<(), Box<::std::error::Error>> {
        let train = include_str!("../data/regression.train");
        let train = parse_csv(train, "\t")?;
        let test = include_str!("../data/regression.test");
        let test = parse_csv(test, "\t")?;

        let loss = RegLoss::default();
        let train = train.as_prepared_data(3_000)?;

        let mut predictions = vec![::std::f64::NAN; train.n_rows()];
        let mut params = TreeParams::new();
        params.max_depth = 6;

        let tree = DecisionTree::build(&train, &mut predictions, &params, loss)?;
        let pred2 = tree.par_predict(&train.features);
        assert_eq!(predictions.len(), pred2.len());

        let diffs: Vec<_> = predictions
            .iter()
            .zip(&pred2)
            .filter(|(&a, &b)| a != b)
            .collect();
        assert_eq!(
            diffs.len(),
            0,
            "{} different predictions out of {}, eg. {:?}",
            diffs.len(),
            predictions.len(),
            diffs.into_iter().take(100).collect::<Vec<_>>()
        );

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
        Ok(())
    }

    #[test]
    fn test_regression_bins_recursive() -> Result<(), Box<::std::error::Error>> {
        let train = include_str!("../data/regression.train");
        let train = parse_csv(train, "\t")?;
        let test = include_str!("../data/regression.test");
        let test = parse_csv(test, "\t")?;

        let loss = RegLoss::default();
        let train = train.as_prepared_data(3_000)?;

        let mut predictions = vec![::std::f64::NAN; train.n_rows()];
        let mut params = TreeParams::new();
        params.max_depth = 6;
        params.recursive_split = true;

        let tree = DecisionTree::build(&train, &mut predictions, &params, loss)?;
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
        Ok(())
    }

    #[test]
    fn test_regression_direct() -> Result<(), Box<::std::error::Error>> {
        let train = include_str!("../data/regression.train");
        let train = parse_csv(train, "\t")?;
        let test = include_str!("../data/regression.test");
        let test = parse_csv(test, "\t")?;

        let loss = RegLoss::default();
        let train = train.as_prepared_data(0)?;

        let mut predictions = vec![::std::f64::NAN; train.n_rows()];
        let mut params = TreeParams::new();
        params.max_depth = 6;

        let tree = DecisionTree::build(&train, &mut predictions, &params, loss)?;
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
        Ok(())
    }
}
