use crate::{Loss, Node, PreparedDataSet, StridedVecView, TreeParams, DEFAULT_N_TREES};
use rand::prelude::Rng;
use rayon::prelude::*;
use std::f64::NAN;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RFParams {
    pub n_trees: usize,
}

impl RFParams {
    pub fn new() -> Self {
        Self {
            n_trees: DEFAULT_N_TREES,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RandomForest<L: Loss> {
    models: Vec<Node>,
    rf_params: RFParams,
    tree_params: TreeParams,
    loss: L,
}

impl<L: Loss + std::marker::Sync> RandomForest<L> {
    pub fn build(
        train: &PreparedDataSet,
        rf_params: &RFParams,
        tree_params: &TreeParams,
        loss: L,
        rng: &mut impl Rng,
    ) -> RandomForest<L> {
        // We have to compute the weights first because they depends on &mut rng
        let weights: Vec<Vec<f64>> = (0..rf_params.n_trees)
            .map(|_| {
                let mut weights: Vec<_> = train.target.iter().map(|_| 0.).collect();
                let indices: Vec<_> = (0..train.target.len()).collect();
                for _ in train.target {
                    weights[*rng.choose(&indices).unwrap()] += 1.;
                }
                weights
            }).collect();

        // We don't do boosting so the initial value is just the default one
        let train_scores: Vec<_> = train.target.iter().map(|_| 0.).collect();

        let models: Vec<_> = weights
            .par_iter()
            .map(|sample_weights| {
                // TODO: can we remove the allocations inside the hot loop?

                // We update the data set to a train set according to the weights.
                let mut train = train.as_train_data(&loss);
                train.update_grad_hessian(&loss, &train_scores, &sample_weights);
                let mut tree_predictions: Vec<_> = train.target.iter().map(|_| NAN).collect();
                let mut cache = Vec::new();

                // We filter the indices with non-null weights
                let indices: Vec<usize> = sample_weights
                    .par_iter()
                    .enumerate()
                    .filter(|(_indice, &weight)| weight > 0.)
                    .map(|(indice, _)| indice)
                    .collect();

                // Let's build it!
                Node::build_from_train_data(
                    &train,
                    &indices,
                    &mut tree_predictions,
                    &tree_params,
                    &mut cache,
                )
            }).collect();

        RandomForest {
            models,
            rf_params: (*rf_params).clone(),
            tree_params: (*tree_params).clone(),
            loss,
        }
    }

    pub fn predict(&self, features: &StridedVecView<f64>) -> f64 {
        let o: f64 = self
            .models
            .iter()
            .map(|model| model.predict(features))
            .sum();
        if o.is_nan() {
            panic!("NAN in output of prediction");
        }
        o / self.models.len() as f64
    }
}
