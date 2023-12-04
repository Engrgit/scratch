#This a code implmenetation of the optimisation of catboost with firefly

import catboost
from catboost import CatBoostClassifier
from hyperopt import fmin, tpe, hp

# Define the search space for hyperparameters
space = {
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
    'depth': hp.choice('depth', range(5, 15)),
    'iterations': hp.choice('iterations', range(50, 200)),
    'l2_leaf_reg': hp.uniform('l2_leaf_reg', 0, 10),
}

# Objective function to minimize (could be accuracy, log loss, etc.)
def objective(params):
    # Convert integer parameters to integer type
    params['depth'] = int(params['depth'])
    params['iterations'] = int(params['iterations'])
    
    # Create and train the CatBoost model with the current set of hyperparameters
    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=10, verbose=False)

    # Define your evaluation metric (for example, log loss)
    metric = model.get_best_score()['validation_0']['Logloss']

    return metric

# Define the number of iterations for the optimization algorithm
max_evals = 20

# Run the optimization with the Firefly Algorithm
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=max_evals, rstate=np.random.RandomState(42))

# Print the best hyperparameters
print("Best Hyperparameters:", best)
