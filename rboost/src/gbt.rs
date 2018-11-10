use std::time::Instant;

use crate::math::sample_indices_ratio;
use crate::{
    cosine_simularity, min_diff_vectors, Dataset, Loss, Node, PreparedDataSet, StridedVecView,
    TrainDataSet, TreeParams, DEFAULT_COLSAMPLE_BYTREE, DEFAULT_LEARNING_RATE,
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

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GBT<L: Loss> {
    models: Vec<Node>,
    booster_params: BoosterParams,
    tree_params: TreeParams,
    best_iteration: usize,
    loss: L,
}

impl<L: Loss> GBT<L> {
    pub fn build(
        booster_params: &BoosterParams,
        tree_params: &TreeParams,
        train_set: &mut PreparedDataSet,
        num_boost_round: usize,
        valid_set: Option<&Dataset>,
        early_stopping_rounds: usize,
        loss: L,
        rng: &mut impl Rng,
    ) -> GBT<L> {
        let mut o = GBT {
            models: Vec::new(),
            booster_params: (*booster_params).clone(),
            tree_params: (*tree_params).clone(),
            best_iteration: 0,
            loss,
        };
        o.train(
            &mut train_set.as_train_data(&o.loss),
            num_boost_round,
            valid_set,
            early_stopping_rounds,
            rng,
        );
        o
    }

    fn train(
        &mut self,
        train: &mut TrainDataSet,
        num_boost_round: usize,
        valid: Option<&Dataset>,
        early_stopping_rounds: usize,
        rng: &mut impl Rng,
    ) {
        // Check we have no NAN in input
        for &x in train.features.flat() {
            if x.is_nan() {
                panic!("Found NAN in the features")
            }
        }
        for &x in train.target {
            if x.is_nan() {
                panic!("Found NAN in the features")
            }
        }

        let mut shrinkage_rate = 1.;
        let mut best_iteration = 0;
        let mut best_val_loss = None;
        let train_start_time = Instant::now();
        println!(
            "Features sorted in {:.03}s",
            train_start_time.elapsed().as_nanos() as f64 / 1_000_000_000.
        );

        println!(
            "Training until validation scores don't improve for {} rounds.",
            early_stopping_rounds
        );
        let mut train_scores: Vec<f64> = (0..train.target.len()).map(|_| 0.).collect();
        let mut val_scores: Option<Vec<f64>> =
            valid.map(|dataset| (0..dataset.target.len()).map(|_| 0.).collect());

        // Predictions per tree. We create it before  so we don't have to allocate a new vector at
        // each iteration
        let mut tree_predictions: Vec<_> = (0..train.target.len()).map(|_| 0.).collect();
        let indices: Vec<usize> = (0..train.target.len()).collect();
        let sample_weights: Vec<_> = (0..train.target.len()).map(|_| 1.).collect();

        for iter_cnt in 0..(num_boost_round) {
            train.update_grad_hessian(&self.loss, &train_scores, &sample_weights);

            if self.booster_params.colsample_bytree < 1. {
                train.columns = sample_indices_ratio(
                    rng,
                    train.features.n_cols(),
                    self.booster_params.colsample_bytree,
                );
            }

            let mut learner = Node::build_from_train_data(
                &train,
                &indices,
                &mut tree_predictions,
                &self.tree_params,
            );
            let alpha = match self.booster_params.booster {
                Booster::CosineSimilarity => {
                    shrinkage_rate * cosine_simularity(&train.grad, &tree_predictions)
                }
                Booster::Geometric => shrinkage_rate,
                Booster::MinDiffVectors => {
                    shrinkage_rate * min_diff_vectors(&train.grad, &tree_predictions)
                }
            };
            learner.apply_shrinking(alpha);

            shrinkage_rate *= self.booster_params.learning_rate;
            for (i, val) in learner.par_predict(&train.features).into_iter().enumerate() {
                train_scores[i] += val;
            }

            if let Some(valid) = valid {
                let val_scores = val_scores.as_mut().expect("No val score");
                for (i, val) in learner.par_predict(&valid.features).into_iter().enumerate() {
                    val_scores[i] += val
                }
                let val_loss =
                    self.loss.calc_loss(&valid.target, &val_scores) / (valid.target.len() as f64);
                let val_loss = val_loss.sqrt();

                if iter_cnt == 0 {
                    best_val_loss = Some(val_loss)
                } else if let Some(best_val_loss_) = best_val_loss {
                    if val_loss < best_val_loss_ {
                        best_val_loss = Some(val_loss);
                        best_iteration = iter_cnt;
                    }
                    if iter_cnt - best_iteration >= early_stopping_rounds {
                        println!("Early stopping, best iteration is:");
                        println!(
                            "Iter {:}, Train's L2: {:.10}",
                            best_iteration,
                            best_val_loss
                                .map(|e| format!("{:.10}", e))
                                .unwrap_or("-".to_string())
                        );
                        for _ in best_iteration..(iter_cnt - 1) {
                            self.models.pop();
                        }
                        break;
                    }
                }
                self.models.push(learner);
            };
        }

        self.best_iteration = best_iteration;
        println!(
            "Training of {} trees finished. Elapsed: {:.2} secs",
            self.models.len(),
            train_start_time.elapsed().as_nanos() as f64 / 1_000_000_000.
        );
    }

    fn _predict(&self, features: &StridedVecView<f64>, models: &[Node]) -> f64 {
        let o: f64 = models.iter().map(|model| model.predict(features)).sum();
        if o.is_nan() {
            panic!("NAN in output of prediction");
        }
        o
    }

    pub fn predict(&self, features: &StridedVecView<f64>) -> f64 {
        self._predict(&features, &self.models)
    }
}
