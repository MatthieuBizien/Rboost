use crate::math::sample_indices_ratio;
use crate::{
    ColumnMajorMatrix, Dataset, FitResult, Loss, Node, PreparedDataset, StridedVecView, TreeParams,
    DEFAULT_COLSAMPLE_BYTREE, SHOULD_NOT_HAPPEN,
};
use rand::prelude::Rng;
use rayon::current_num_threads;
use rayon::prelude::*;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DartParams {
    pub colsample_bytree: f64,
    pub dropout_rate: f64,
    pub learning_rate: f64,
}

impl DartParams {
    pub fn new() -> Self {
        DartParams {
            colsample_bytree: DEFAULT_COLSAMPLE_BYTREE,
            dropout_rate: 0.5,
            learning_rate: 1.0,
        }
    }
}

impl Default for DartParams {
    fn default() -> Self {
        Self::new()
    }
}

/// Dart Booster. Boosting with dropout.
///
/// Based on <https://arxiv.org/abs/1505.01866>
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Dart<L: Loss> {
    models: Vec<Node>,
    booster_params: DartParams,
    tree_params: TreeParams,
    best_iteration: usize,
    loss: L,
    initial_prediction: f64,
}

/// Compute the prediction for multiple trees of a DART
fn calc_full_prediction(
    predictions: &ColumnMajorMatrix<f64>,
    tree_weights: &[f64],
    filter: &[bool],
    dest: &mut [f64],
    learning_rate: f64,
    initial_prediction: f64,
) {
    let chunk_size = predictions.n_rows() / current_num_threads();
    // We split the dataset by rows
    dest.par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(n_chunk, dest)| {
            // Initialization of the dataset at the initial prediction
            dest.iter_mut().for_each(|e| *e = initial_prediction);

            let first_original_idx = chunk_size * n_chunk;

            // We add the trees to the current chunk
            for ((predictions, &weight), &keep) in
                predictions.columns().zip(tree_weights).zip(filter)
            {
                // Dropout: we only keep parts of the trees
                if !keep {
                    continue;
                }
                let predictions = &predictions[first_original_idx..];
                for (dest, prediction) in dest.iter_mut().zip(predictions) {
                    *dest += prediction * weight * learning_rate;
                }
            }
        })
}

impl<L: Loss> Dart<L> {
    pub fn build(
        booster_params: &DartParams,
        tree_params: &TreeParams,
        train_set: &mut PreparedDataset,
        num_boost_round: usize,
        valid: Option<&Dataset>,
        early_stopping_rounds: usize,
        loss: L,
        rng: &mut impl Rng,
    ) -> FitResult<Dart<L>> {
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
        let mut active_tree = vec![true; num_boost_round];

        for iter_cnt in 0..(num_boost_round) {
            // Skip some trees for predictions
            for e in active_tree.iter_mut().take(iter_cnt) {
                if rng.gen::<f64>() > booster_params.dropout_rate {
                    *e = false;
                } else {
                    *e = true
                }
            }
            if active_tree.iter().all(|&e| e) & (booster_params.dropout_rate < 1.0) {
                *rng.choose_mut(&mut active_tree).expect(SHOULD_NOT_HAPPEN) = true;
            }
            let n_removed: usize = active_tree.iter().map(|&e| (!e) as usize).sum();
            let normalizer = (n_removed as f64) / (1. + n_removed as f64);
            for (&e, weight) in active_tree.iter().zip(tree_weights.iter_mut()) {
                if !e {
                    *weight *= normalizer;
                }
            }

            calc_full_prediction(
                &train_preds,
                &tree_weights,
                &active_tree,
                &mut train_scores,
                booster_params.learning_rate,
                initial_prediction,
            );
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

            tree_weights.push(1. / (1. + n_removed as f64));

            if let Some(valid) = valid {
                let val_predictions = val_predictions.as_mut().expect(SHOULD_NOT_HAPPEN);
                let mut val_scores = val_scores.as_mut().expect(SHOULD_NOT_HAPPEN);
                let dest = val_predictions.column_mut(iter_cnt);
                dest.clone_from_slice(&learner.par_predict(&valid.features));
                active_tree.iter_mut().for_each(|e| *e = true);
                calc_full_prediction(
                    &val_predictions,
                    &tree_weights,
                    &active_tree,
                    &mut val_scores,
                    1.,
                    initial_prediction,
                );
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
