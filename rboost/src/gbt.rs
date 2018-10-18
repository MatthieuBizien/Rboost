use std::time::Instant;

use crate::{Dataset, Loss, Node, Params, RegLoss, StridedVecView, TrainDataSet};

pub struct GBT {
    models: Vec<Node>,
    params: Params,
    best_iteration: usize,
    loss: Box<RegLoss>,
}

impl GBT {
    fn _build_learner(&self, train: &TrainDataSet, shrinkage_rate: f64) -> Node {
        let depth = 0;
        let indices: Vec<usize> = (0..train.target.len()).collect();
        Node::build(&train, &indices, shrinkage_rate, depth, &self.params)
    }

    pub fn build(
        params: &Params,
        train_set: &Dataset,
        num_boost_round: usize,
        valid_set: Option<&Dataset>,
        early_stopping_rounds: usize,
    ) -> GBT {
        let mut o = GBT {
            models: Vec::new(),
            params: (*params).clone(),
            best_iteration: 0,
            loss: Box::new(RegLoss::default()),
        };
        o.train(train_set, num_boost_round, valid_set, early_stopping_rounds);
        o
    }

    fn train(
        &mut self,
        train_set: &Dataset,
        num_boost_round: usize,
        valid_set: Option<&Dataset>,
        early_stopping_rounds: usize,
    ) {
        // Check we have no NAN in input
        for &x in train_set.features.flat() {
            if x.is_nan() {
                panic!("Found NAN in the features")
            }
        }
        for &x in &train_set.target {
            if x.is_nan() {
                panic!("Found NAN in the features")
            }
        }

        let mut shrinkage_rate = 1.;
        let mut best_iteration = 0;
        let mut best_val_loss = None;
        let train_start_time = Instant::now();

        let sorted_features = train_set.sort_features();
        let (bins, n_bins) = Dataset::bin_features(&sorted_features, self.params.n_bins);
        println!(
            "Features sorted in {:.03}s",
            train_start_time.elapsed().as_nanos() as f64 / 1_000_000_000.
        );

        println!(
            "Training until validation scores don't improve for {} rounds.",
            early_stopping_rounds
        );
        let mut train_scores: Vec<f64> = (0..train_set.target.len()).map(|_| 0.).collect();
        let mut val_scores: Option<Vec<f64>> =
            valid_set.map(|dataset| (0..dataset.target.len()).map(|_| 0.).collect());

        for iter_cnt in 0..(num_boost_round) {
            let (grad, hessian) = self.loss.calc_gradient(&train_set.target, &train_scores);

            // TODO seems to work, but is it sound?
            let grad = if iter_cnt == 0 {
                grad.into_iter().map(|i| i * 0.5).collect()
            } else {
                grad
            };

            let train = TrainDataSet {
                features: &train_set.features,
                target: &train_set.target,
                sorted_features: &sorted_features,
                grad,
                hessian,
                bins: &bins,
                n_bins: &n_bins,
            };
            let learner = self._build_learner(&train, shrinkage_rate);
            if iter_cnt > 0 {
                shrinkage_rate *= self.params.learning_rate;
            }
            for (i, val) in learner.par_predict(&train_set).into_iter().enumerate() {
                train_scores[i] += val;
            }

            if let Some(valid_set) = valid_set {
                let val_scores = val_scores.as_mut().expect("No val score");
                for (i, val) in learner.par_predict(&valid_set).into_iter().enumerate() {
                    val_scores[i] += val
                }
                let val_loss = self.loss.calc_loss(&valid_set.target, &val_scores);

                if let Some(best_val_loss_) = best_val_loss {
                    if val_loss < best_val_loss_ || iter_cnt == 0 {
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
                        break;
                    }
                }
                self.models.push(learner);
            };
        }

        self.best_iteration = best_iteration;
        println!(
            "Training finished. Elapsed: {:.2} secs",
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
