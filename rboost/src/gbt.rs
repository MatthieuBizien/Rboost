use crate::math::sample_indices_ratio;
use crate::{
    cosine_simularity, min_diff_vectors, ColumnMajorMatrix, Dataset, FitResult, Loss, Node,
    PreparedDataset, StridedVecView, TreeParams, DEFAULT_COLSAMPLE_BYTREE, DEFAULT_LEARNING_RATE,
    SHOULD_NOT_HAPPEN,
};
use rand::prelude::Rng;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum Booster {
    Geometric,
    CosineSimilarity,
    MinDiffVectors,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct BoosterParams {
    pub learning_rate: f64,
    pub booster: Booster,
    pub colsample_bytree: f64,
}

impl BoosterParams {
    pub fn new() -> Self {
        BoosterParams {
            learning_rate: DEFAULT_LEARNING_RATE,
            booster: Booster::Geometric,
            colsample_bytree: DEFAULT_COLSAMPLE_BYTREE,
        }
    }
}

impl Default for BoosterParams {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GBT<L: Loss> {
    models: Vec<Node>,
    booster_params: BoosterParams,
    tree_params: TreeParams,
    best_iteration: usize,
    loss: L,
}

/// Compute the prediction for multiple trees of a GBT.
fn calc_full_prediction(
    predictions: &ColumnMajorMatrix<f64>,
    tree_weights: &[f64],
    dest: &mut [f64],
) {
    dest.iter_mut().for_each(|e| *e = 0.);
    for (predictions, &weight) in predictions.columns().zip(tree_weights) {
        for (dest, prediction) in dest.iter_mut().zip(predictions) {
            *dest += prediction * weight;
        }
    }
}

impl<L: Loss> GBT<L> {
    pub fn build(
        booster_params: &BoosterParams,
        tree_params: &TreeParams,
        train_set: &mut PreparedDataset,
        num_boost_round: usize,
        valid: Option<&Dataset>,
        early_stopping_rounds: usize,
        loss: L,
        rng: &mut impl Rng,
    ) -> FitResult<GBT<L>> {
        let mut models = Vec::new();
        let booster_params = (*booster_params).clone();
        let tree_params = (*tree_params).clone();
        let mut train = train_set.as_train_data(&loss);

        train.check_data()?;

        let mut shrinkage_rate = 1.;
        let mut best_iteration = 0;
        let mut best_val_loss = None;
        let size_train = train.target.len();
        let size_val = valid.map(|e| e.target.len());

        // Predictions per tree.
        let mut train_preds =
            ColumnMajorMatrix::from_function(size_train, num_boost_round, |_, _| 0.);
        let mut val_predictions = size_val
            .map(|size_val| ColumnMajorMatrix::from_function(size_val, num_boost_round, |_, _| 0.));

        // Weights of every trees
        let mut tree_weights = Vec::new();

        // Indices and weights per tree. We create it before  so we don't have to allocate a new vector at
        // each iteration
        let indices: Vec<usize> = (0..train.target.len()).collect();
        let sample_weights = vec![1.; size_train];
        let mut train_scores = vec![0.; size_train];
        let mut val_scores = size_val.map(|size_val| vec![0.; size_val]);

        for iter_cnt in 0..(num_boost_round) {
            calc_full_prediction(&train_preds, &tree_weights, &mut train_scores);
            train.update_grad_hessian(&loss, &train_scores, &sample_weights);

            if booster_params.colsample_bytree < 1. {
                train.columns = sample_indices_ratio(
                    rng,
                    train.features.n_cols(),
                    booster_params.colsample_bytree,
                );
            }

            let learner = Node::build_from_train_data(
                &train,
                &indices,
                train_preds.column_mut(iter_cnt),
                &tree_params,
            );

            tree_weights.push(match booster_params.booster {
                Booster::CosineSimilarity => {
                    shrinkage_rate * cosine_simularity(&train.grad, &train_preds.column(iter_cnt))
                }
                Booster::Geometric => shrinkage_rate,
                Booster::MinDiffVectors => {
                    shrinkage_rate * min_diff_vectors(&train.grad, &train_preds.column(iter_cnt))
                }
            });
            shrinkage_rate *= booster_params.learning_rate;

            if let Some(valid) = valid {
                let val_predictions = val_predictions.as_mut().expect(SHOULD_NOT_HAPPEN);
                let mut val_scores = val_scores.as_mut().expect(SHOULD_NOT_HAPPEN);
                let dest = val_predictions.column_mut(iter_cnt);
                dest.clone_from_slice(&learner.par_predict(&valid.features));
                calc_full_prediction(&val_predictions, &tree_weights, &mut val_scores);
                let val_loss =
                    loss.calc_loss(&valid.target, &val_scores) / (valid.target.len() as f64);
                let val_loss = val_loss.sqrt();

                if iter_cnt == 0 {
                    best_val_loss = Some(val_loss)
                } else if let Some(best_val_loss_) = best_val_loss {
                    if val_loss < best_val_loss_ {
                        best_val_loss = Some(val_loss);
                        best_iteration = iter_cnt;
                    }
                    if iter_cnt - best_iteration >= early_stopping_rounds {
                        for _ in best_iteration..(iter_cnt - 1) {
                            models.pop();
                        }
                        break;
                    }
                }
                models.push(learner);
            };
        }

        for (tree, &weight) in models.iter_mut().zip(&tree_weights) {
            tree.apply_shrinking(weight);
        }
        Ok(Self {
            models,
            booster_params,
            tree_params,
            best_iteration,
            loss,
        })
    }

    fn _predict(&self, features: &StridedVecView<f64>, models: &[Node]) -> f64 {
        let o: f64 = models.iter().map(|model| model.predict(features)).sum();
        if o.is_nan() {
            panic!("NAN in output of prediction");
        }
        self.loss.get_target(o)
    }

    pub fn predict(&self, features: &StridedVecView<f64>) -> f64 {
        self._predict(&features, &self.models)
    }

    pub fn n_trees(&self) -> usize {
        self.models.len()
    }
}
