import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
from data_preprocessing import load_and_preprocess_data


def train_model(model, model_name, X_train, X_test, y_train, y_test):
    """Train model and log with MLflow"""

    with mlflow.start_run(run_name=f"{model_name}_experiment"):
        # Train model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred_test = model.predict(X_test)

        # Calculate metrics
        test_mse = mean_squared_error(y_test, y_pred_test)
        test_r2 = r2_score(y_test, y_pred_test)
        test_mae = mean_absolute_error(y_test, y_pred_test)

        # Log parameters
        if hasattr(model, 'n_estimators'):
            mlflow.log_param("n_estimators", model.n_estimators)
        if hasattr(model, 'max_depth'):
            mlflow.log_param("max_depth", model.max_depth)

        # Log metrics
        mlflow.log_metric("test_mse", test_mse)
        mlflow.log_metric("test_r2", test_r2)
        mlflow.log_metric("test_mae", test_mae)

        # Log model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=f"housing_{model_name.lower()}"
        )

        print(f"{model_name} - MSE: {test_mse:.4f}, R2: {test_r2:.4f}")

        return model, test_mse, test_r2


def main():
    """Main training pipeline"""

    # Set MLflow tracking
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment("housing_price_prediction")

    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data()

    # Define models
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "DecisionTree": DecisionTreeRegressor(random_state=42, max_depth=10)
    }

    best_model = None
    best_mse = float('inf')
    best_model_name = ""

    print("Training models...")

    # Train and evaluate models
    for name, model in models.items():
        print(f"Training {name}...")
        trained_model, mse, r2 = train_model(model, name, X_train, X_test, y_train, y_test)

        if mse < best_mse:
            best_mse = mse
            best_model = trained_model
            best_model_name = name

    # Save best model
    best_model_filename = f'models/best_model_{best_model_name.lower()}.pkl'
    joblib.dump(best_model, best_model_filename)

    print(f"\nBest model: {best_model_name} with MSE: {best_mse:.4f}")
    print(f"Model saved as: {best_model_filename}")

    return best_model, best_model_name


if __name__ == "__main__":
    main()
