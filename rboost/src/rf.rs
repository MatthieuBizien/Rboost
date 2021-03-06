use crate::math::sample_indices_ratio;
use crate::{
    FitResult, Loss, Node, PreparedDataset, StridedVecView, TreeParams, DEFAULT_COLSAMPLE_BYTREE,
    DEFAULT_N_TREES, SHOULD_NOT_HAPPEN,
};
use rand::prelude::Rng;
use rayon::prelude::*;
use std::sync::Arc;
use std::sync::Mutex;

/// Parameters for constructing a random forest.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RFParams {
    pub n_trees: usize,
    pub colsample_bytree: f64,
}

impl RFParams {
    pub fn new() -> Self {
        Self {
            n_trees: DEFAULT_N_TREES,
            colsample_bytree: DEFAULT_COLSAMPLE_BYTREE,
        }
    }
}

impl Default for RFParams {
    fn default() -> Self {
        Self::new()
    }
}

/// Random Forest implementation.
///
/// The trees are constructed independently by sub-sampling with resample.
/// It's parallelized using rayon.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RandomForest<L: Loss> {
    models: Vec<Node>,
    rf_params: RFParams,
    tree_params: TreeParams,
    loss: L,
    initial_prediction: f64,
}

impl<L: Loss + std::marker::Sync> RandomForest<L> {
    pub fn build(
        train: &PreparedDataset,
        rf_params: &RFParams,
        tree_params: &TreeParams,
        loss: L,
        rng: &mut impl Rng,
    ) -> FitResult<(RandomForest<L>, Vec<f64>)> {
        train.check_data()?;
        let initial_prediction = loss.get_initial_prediction(&train.target);

        // We have to compute the weights first because they depends on &mut rng
        let random_init: Vec<_> = (0..rf_params.n_trees)
            .map(|_| {
                let mut weights = vec![0.; train.n_rows()];
                let indices: Vec<_> = (0..train.target.len()).collect();
                for _ in train.target {
                    weights[*rng.choose(&indices).expect(SHOULD_NOT_HAPPEN)] += 1.;
                }
                let columns =
                    sample_indices_ratio(rng, train.features.n_cols(), rf_params.colsample_bytree);
                (weights, columns)
            })
            .collect();

        // We don't do boosting so the initial value is just the default one
        let train_scores = vec![initial_prediction; train.n_rows()];

        // Store the sum of the cross-validated predictions and the number of predictions done
        let predictions = vec![(0., 0); train.n_rows()];
        let predictions = Arc::new(Mutex::new(predictions));

        let models: Vec<_> = random_init
            .into_par_iter()
            .map(|(sample_weights, columns)| {
                // TODO: can we remove the allocations inside the hot loop?

                // We update the data set to a train set according to the weights.
                let mut train = train.as_train_data(&loss);
                train.update_grad_hessian(&loss, &train_scores, &sample_weights);
                train.columns = columns;

                let mut tree_predictions = vec![0.; train.n_rows()];

                // We filter the indices with non-null weights
                let (mut train_indices, mut test_indices) = (Vec::new(), Vec::new());
                for (indice, &weight) in sample_weights.iter().enumerate() {
                    if weight > 0. {
                        train_indices.push(indice)
                    } else {
                        test_indices.push(indice)
                    }
                }

                // Let's build it!
                let node = Node::build_from_train_data(
                    &train,
                    &train_indices,
                    &mut tree_predictions,
                    &tree_params,
                );

                // Predict and write back the predictions to the result
                let predictions = predictions.clone();
                let mut predictions = predictions.lock().expect("Poisoned mutex");
                for i in test_indices {
                    predictions[i].0 += node.predict(&train.features.row(i));
                    predictions[i].1 += 1;
                }
                node
            })
            .collect();

        // We divide the cross-validated predictions by the number of trees used
        let predictions = predictions.lock().expect("Poisoned mutex");
        let predictions = predictions
            .iter()
            .map(|(pred, n)| *pred / (*n as f64))
            .map(|latent| loss.get_target(latent + initial_prediction))
            .collect();

        let rf = RandomForest {
            models,
            rf_params: (*rf_params).clone(),
            tree_params: (*tree_params).clone(),
            loss,
            initial_prediction,
        };

        Ok((rf, predictions))
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
        let o = o / self.models.len() as f64;
        self.loss.get_target(o + self.initial_prediction)
    }
}
