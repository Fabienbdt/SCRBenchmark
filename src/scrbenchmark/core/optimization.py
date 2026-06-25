
"""
Hyperparameter optimization module using Optuna.
Supports Bayesian optimization (TPE) and pruning strategies.
"""

import logging
import optuna
import numpy as np
from typing import Any, Dict, List, Optional, Union, Callable
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import streamlit as st

from .algorithm_registry import AlgorithmRegistry, BaseAlgorithm
from .config import HyperparameterConfig, ParamType

logger = logging.getLogger(__name__)

class OptunaOptimizer:
    """
    Hyperparameter optimizer using Optuna.
    """
    
    def __init__(self, 
                 algorithm_name: str,
                 study_name: str = None,
                 direction: str = "maximize",
                 storage: str = None):
        """
        Initialize the optimizer.
        
        Args:
            algorithm_name: Name of the algorithm to optimize
            study_name: Name of the optuna study
            direction: 'maximize' or 'minimize'
            storage: Database URL for storage (optional)
        """
        self.algorithm_name = algorithm_name
        self.direction = direction
        self.storage = storage
        
        # Create study
        self.study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            storage=storage,
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=42),  # Tree-structured Parzen Estimator
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
        
    def optimize(self, 
                 train_data: Any,
                 val_data: Any, 
                 param_space: Dict[str, List[Any]],
                 n_trials: int = 20,
                 metric: str = 'nmi',
                 labels_val: Optional[Any] = None,
                 callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Run the optimization process.
        
        Args:
            train_data: Data for training
            val_data: Data for validation
            param_space: Dictionary defining the search space.
                         Format: {'param_name': [list_of_values]} or
                         {'param_name': {'min': x, 'max': y, 'type': 'int/float'}}
            n_trials: Number of trials to run
            metric: Metric to optimize ('nmi', 'ari', 'accuracy')
            labels_val: Ground truth labels for validation
            callback: Optional callback to update UI progress
            
        Returns:
            Best parameters found
        """
        if labels_val is None:
            raise ValueError("Validation labels are required for supervised metric evaluation.")
            
        def objective(trial):
            # 1. Sample hyperparameters
            params = {}
            for param_name, space in param_space.items():
                if isinstance(space, list):
                    # Categorical choices
                    params[param_name] = trial.suggest_categorical(param_name, space)
                elif isinstance(space, dict):
                    # Range
                    p_type = space.get('type', 'float')
                    if p_type == 'int':
                        step = space.get('step', 1)
                        if isinstance(step, float): # Create integer list if step is float (legacy compat)
                             params[param_name] = trial.suggest_int(param_name, space['min'], space['max'])
                        else:
                             params[param_name] = trial.suggest_int(param_name, space['min'], space['max'], step=step)
                    elif p_type == 'float':
                        log = space.get('log', False)
                        params[param_name] = trial.suggest_float(
                            param_name, 
                            space['min'], 
                            space['max'], 
                            step=space.get('step', None) if not log else None, 
                            log=log
                        )
                    elif p_type == 'bool':
                        params[param_name] = trial.suggest_categorical(param_name, [True, False])

            # 2. Instantiate algorithm
            algo_class = AlgorithmRegistry.get(self.algorithm_name)
            if not algo_class:
                raise ValueError(f"Algorithm {self.algorithm_name} not found")
            
            # Inject trial for pruning support (algorithms can check params['optuna_trial'])
            params['optuna_trial'] = trial
            params['verbose'] = False # Reduce logging during search
            
            model = algo_class(params)
            
            # 3. Fit and Predict
            try:
                # Check if algorithm supports out-of-sample prediction
                info = algo_class.get_info()
                supports_oos = getattr(info, 'supports_out_of_sample', True) # Default to True if not specified
                
                if supports_oos:
                    # Standard Inductive case: Fit on Train, Predict on Val
                    model.fit(train_data)
                    
                    if hasattr(model, 'predict_new'): 
                         y_pred = model.predict_new(val_data)
                    else:
                        try:
                            # Try predicting on validation data
                            # Note: some algos might just ignore argument and return train labels (scCDCG behavior)
                            # so we must be careful. But supports_oos=True implies it SHOULD work.
                            y_pred = model.predict(val_data)
                        except Exception as e:
                            logger.warning(f"Inductive predict failed: {e}")
                            raise optuna.TrialPruned()
                else:
                    # Transductive case (e.g. Graph-based): Cannot predict on new data.
                    # To evaluate on Validation Set, we must FIT on Validation Set.
                    # This finds params that work well on the validation distribution.
                    # logger.info(f"Algorithm {self.algorithm_name} is transductive. Fitting on Validation Data.")
                    model.fit(val_data, labels=labels_val)
                    y_pred = model.predict(val_data) # Returns labels for fitted data (val_data)

                # 4. Evaluate
                score = 0
                if metric == 'nmi':
                    score = normalized_mutual_info_score(labels_val, y_pred)
                elif metric == 'ari':
                    score = adjusted_rand_score(labels_val, y_pred)
                else:
                    score = normalized_mutual_info_score(labels_val, y_pred) # Default
                    
                # Update UI callback if provided
                if callback:
                    callback(trial.number, params, score)
                    
                return score

            except optuna.TrialPruned:
                raise
            except Exception as e:
                logger.error(f"Trial {trial.number} failed: {str(e)}")
                # Return a very low score for failures
                return 0.0 if self.direction == 'maximize' else float('inf')

        # Run optimization
        self.study.optimize(
            objective, 
            n_trials=n_trials, 
            callbacks=[], # Can add standard optuna callbacks
            n_jobs=1  # Sequential to avoid GPU OOM
        )
        
        return self.study.best_params

    def get_summary(self):
        """Return summary of the study."""
        return self.study.trials_dataframe()

def get_param_space_from_config(configs: List[HyperparameterConfig]) -> Dict[str, Any]:
    """Helper to convert HyperparameterConfig list to Optuna search space dict."""
    space = {}
    for conf in configs:
        if conf.param_type == ParamType.INTEGER:
            space[conf.name] = {
                'min': conf.min_value, 
                'max': conf.max_value, 
                'type': 'int', 
                'step': conf.step
            }
        elif conf.param_type == ParamType.FLOAT:
            # Check for log scale heuristic
            log = False
            if conf.min_value is not None and conf.min_value > 0:
                if (conf.max_value / conf.min_value) >= 100:
                    log = True
            
            space[conf.name] = {
                'min': conf.min_value, 
                'max': conf.max_value, 
                'type': 'float', 
                'step': conf.step,
                'log': log
            }
        elif conf.param_type == ParamType.CHOICE:
            space[conf.name] = conf.choices
        elif conf.param_type == ParamType.BOOLEAN:
            space[conf.name] = [True, False]
            
    return space
