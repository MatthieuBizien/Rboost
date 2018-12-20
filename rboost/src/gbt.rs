use crate::math::add;
use crate::math::mul_add;
use crate::math::sample_indices_ratio;
use crate::{
    Dataset, FitResult, Loss, Node, PreparedDataset, StridedVecView, TreeParams,
    DEFAULT_COLSAMPLE_BYTREE, DEFAULT_LEARNING_RATE, SHOULD_NOT_HAPPEN,
};
use rand::prelude::Rng;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct BoosterParams {
    pub learning_rate: f64,
    pub colsample_bytree: f64,
}

impl BoosterParams {
    pub fn new() -> Self {
        BoosterParams {
            learning_rate: DEFAULT_LEARNING_RATE,
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
    initial_prediction: f64,
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

        let initial_prediction = loss.get_initial_prediction(&train.target);

        let mut best_iteration = 0;
        let mut best_val_loss = None;
        let size_train = train.target.len();
        let size_val = valid.map(|e| e.target.len());

        // Indices and weights per tree. We create it before  so we don't have to allocate a new vector at
        // each iteration
        let indices: Vec<usize> = (0..train.target.len()).collect();
        let sample_weights = vec![1.; size_train];
        let mut train_scores = vec![initial_prediction; size_train];
        let mut val_scores = size_val.map(|size_val| vec![initial_prediction; size_val]);

        let mut train_cache_score = vec![0.; size_train];
        let mut val_cache_score = size_val.map(|size_val| vec![0.; size_val]);

        for iter_cnt in 0..(num_boost_round) {
            train.update_grad_hessian(&loss, &train_scores, &sample_weights);

            if booster_params.colsample_bytree < 1. {
                train.columns = sample_indices_ratio(
                    rng,
                    train.features.n_cols(),
                    booster_params.colsample_bytree,
                );
            }

            let mut learner =
                Node::build_from_train_data(&train, &indices, &mut train_cache_score, &tree_params);

            mul_add(
                &train_cache_score,
                booster_params.learning_rate,
                &mut train_scores,
            );
            learner.apply_shrinking(booster_params.learning_rate);

            if let Some(valid) = valid {
                let val_cache_score = val_cache_score.as_mut().expect(SHOULD_NOT_HAPPEN);
                val_cache_score.clone_from_slice(&learner.par_predict(&valid.features));
                let val_scores = val_scores.as_mut().expect(SHOULD_NOT_HAPPEN);
                add(&val_cache_score, val_scores);
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

        Ok(Self {
            models,
            booster_params,
            tree_params,
            best_iteration,
            loss,
            initial_prediction,
        })
    }

    fn _predict(&self, features: &StridedVecView<f64>, models: &[Node]) -> f64 {
        let o: f64 = models.iter().map(|model| model.predict(features)).sum();
        if o.is_nan() {
            panic!("NAN in output of prediction");
        }
        self.loss.get_target(o + self.initial_prediction)
    }

    pub fn predict(&self, features: &StridedVecView<f64>) -> f64 {
        self._predict(&features, &self.models)
    }

    pub fn n_trees(&self) -> usize {
        self.models.len()
    }
}
