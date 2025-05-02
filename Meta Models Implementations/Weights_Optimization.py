import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error


def optimize_ensemble_weights(model_predictions, ground_truth, metric=mean_squared_error):
    """
    Optimizes weights for a weighted average of different model predictions.
    Automatically handles NaN values.
    
    Parameters:
    -----------
    model_predictions : DataFrame
        DataFrame with predictions from each model in the columns
    ground_truth : array-like
        Actual values (target)
    metric : function, default=mean_squared_error
        Metric function to minimize
        
    Returns:
    --------
    weights : array
        Optimized weights for each model
    weighted_predictions : array
        Predictions obtained using the optimized weights
    """
    if not isinstance(model_predictions, pd.DataFrame):
        model_predictions = pd.DataFrame(model_predictions)

  
    def objective(weights):
        weights = weights / np.sum(weights)
        
        weighted_preds = np.zeros(len(ground_truth))
        
        for i in range(len(ground_truth)):
            row = model_predictions.iloc[i].values
            valid_indices = ~np.isnan(row)
            
            if np.any(valid_indices):
                valid_weights = weights[valid_indices]
                valid_preds = row[valid_indices]
                
                if len(valid_weights) > 0:
                    norm_valid_weights = valid_weights / np.sum(valid_weights)
                    weighted_preds[i] = np.sum(norm_valid_weights * valid_preds)
            else:
                weighted_preds[i] = np.nanmean(ground_truth)
        
        valid_indices = ~np.isnan(weighted_preds)
        if np.any(valid_indices):
            return metric(ground_truth[valid_indices], weighted_preds[valid_indices])
        else:
            return float('inf')
    
    n_models = model_predictions.shape[1]
    initial_weights = np.ones(n_models) / n_models

    #constraints: weights has to be nonnegative and has to sum uèp to 1
    bounds = [(0, 1) for * in range(n*models)]
    constraint = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    
    result = minimize(objective, initial_weights, bounds=bounds, constraints=constraint)
    
    optimized_weights = result.x

  
    def get_weighted_prediction(weights, predictions):
        weighted_preds = np.zeros(len(ground_truth))
        
        for i in range(len(ground_truth)):
            row = predictions.iloc[i].values
            valid_indices = ~np.isnan(row)
            
            if np.any(valid_indices):
                valid_weights = weights[valid_indices]
                valid_preds = row[valid_indices]
                
                if len(valid_weights) > 0:
                    norm_valid_weights = valid_weights / np.sum(valid_weights)
                    weighted_preds[i] = np.sum(norm_valid_weights * valid_preds)
            else:
                weighted_preds[i] = np.nan
        
        return weighted_preds
    
    weighted_predictions = get_weighted_prediction(optimized_weights, model_predictions)
    
    return optimized_weights, weighted_predictions
