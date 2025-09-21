import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import io
import base64
from typing import Dict, List, Tuple, Any, Optional
import json

class SprintPerformanceModel:
    """
    A model that generates insights and predictions about sprint running performance
    using data extracted from scientific papers.
    """
    
    def __init__(self):
        """Initialize the model with default parameters"""
        self.models = {}
        self.data_tables = {}
        self.equations = {}
        self.variable_relationships = {}
        
    def add_data_table(self, name: str, data: Dict[str, List[float]]) -> None:
        """
        Add a data table extracted from scientific papers
        
        Args:
            name: Identifier for the data table
            data: Dictionary with column names as keys and lists of values
        """
        try:
            # Convert to pandas DataFrame for easier manipulation
            self.data_tables[name] = pd.DataFrame(data)
            print(f"Added data table '{name}' with columns: {list(data.keys())}")
        except Exception as e:
            print(f"Error adding data table: {str(e)}")
    
    def add_equation(self, name: str, equation_str: str, variables: Dict[str, str]) -> None:
        """
        Add an equation extracted from scientific papers
        
        Args:
            name: Identifier for the equation
            equation_str: String representation of the equation
            variables: Dictionary mapping variable symbols to descriptions
        """
        self.equations[name] = {
            "equation": equation_str,
            "variables": variables
        }
        print(f"Added equation '{name}': {equation_str}")
    
    def train_model(self, model_name: str, data_table_name: str, 
                   target_col: str, feature_cols: List[str],
                   model_type: str = "linear") -> Dict[str, Any]:
        """
        Train a predictive model based on data from scientific papers
        
        Args:
            model_name: Name to identify this model
            data_table_name: Which data table to use
            target_col: Target variable to predict
            feature_cols: List of feature columns
            model_type: Type of model (linear, polynomial, random_forest)
            
        Returns:
            Dictionary with model statistics and performance metrics
        """
        if data_table_name not in self.data_tables:
            return {"error": f"Data table '{data_table_name}' not found"}
            
        data = self.data_tables[data_table_name]
        
        # Ensure all required columns exist
        for col in [target_col] + feature_cols:
            if col not in data.columns:
                return {"error": f"Column '{col}' not found in data table"}
        
        # Extract features and target
        X = data[feature_cols].values
        y = data[target_col].values
        
        # Handle missing values
        if np.isnan(X).any() or np.isnan(y).any():
            # Simple imputation with mean - could be more sophisticated
            for i, col in enumerate(feature_cols):
                mean_val = np.nanmean(X[:, i])
                X[:, i] = np.nan_to_num(X[:, i], nan=mean_val)
            
            mean_y = np.nanmean(y)
            y = np.nan_to_num(y, nan=mean_y)
        
        # Train model based on specified type
        model = None
        
        if model_type == "linear":
            model = LinearRegression()
        elif model_type == "polynomial":
            # 2nd degree polynomial
            poly = PolynomialFeatures(degree=2)
            X_poly = poly.fit_transform(X)
            model = Ridge(alpha=0.1)  # Ridge to prevent overfitting
            X = X_poly
        elif model_type == "random_forest":
            model = RandomForestRegressor(n_estimators=100)
        else:
            return {"error": f"Unknown model type: {model_type}"}
        
        # Train the model
        model.fit(X, y)
        
        # Get predictions and calculate metrics
        y_pred = model.predict(X)
        mse = np.mean((y - y_pred) ** 2)
        r_squared = model.score(X, y)
        
        # Store the model
        self.models[model_name] = {
            "model": model,
            "model_type": model_type,
            "data_table": data_table_name,
            "target": target_col,
            "features": feature_cols,
            "metrics": {
                "mse": mse,
                "r_squared": r_squared
            }
        }
        
        # If linear model, extract coefficients as relationships
        if model_type == "linear":
            relationships = {}
            for i, feature in enumerate(feature_cols):
                relationships[feature] = {
                    "coefficient": float(model.coef_[i]),
                    "impact": "positive" if model.coef_[i] > 0 else "negative",
                    "magnitude": abs(model.coef_[i])
                }
            
            self.variable_relationships[model_name] = relationships
            
        # Return model info
        return {
            "model_name": model_name,
            "model_type": model_type,
            "features": feature_cols,
            "target": target_col,
            "metrics": {
                "mse": float(mse),
                "r_squared": float(r_squared)
            },
            "message": f"Model '{model_name}' trained successfully"
        }
    
    def predict(self, model_name: str, input_values: Dict[str, float]) -> Dict[str, Any]:
        """
        Make a prediction using a trained model
        
        Args:
            model_name: Name of the model to use
            input_values: Dictionary mapping feature names to values
            
        Returns:
            Dictionary with prediction and confidence information
        """
        if model_name not in self.models:
            return {"error": f"Model '{model_name}' not found"}
        
        model_info = self.models[model_name]
        model = model_info["model"]
        features = model_info["features"]
        
        # Check that all features are provided
        for feature in features:
            if feature not in input_values:
                return {"error": f"Missing value for feature '{feature}'"}
        
        # Create input array in correct order
        X = np.array([input_values[f] for f in features]).reshape(1, -1)
        
        # Handle polynomial features if needed
        if model_info["model_type"] == "polynomial":
            poly = PolynomialFeatures(degree=2)
            X = poly.fit_transform(X)
        
        # Make prediction
        prediction = float(model.predict(X)[0])
        
        # Calculate confidence (simplified)
        confidence = 0.95 * model_info["metrics"]["r_squared"]
        
        return {
            "prediction": prediction,
            "target_variable": model_info["target"],
            "confidence": confidence,
            "input_values": input_values
        }
    
    def generate_synthetic_data(self, model_name: str, 
                              feature_ranges: Dict[str, Tuple[float, float, int]],
                              sampling: str = "grid") -> Dict[str, Any]:
        """
        Generate synthetic data points using a trained model
        
        Args:
            model_name: Name of the model to use
            feature_ranges: Dict mapping feature names to (min, max, num_points)
            sampling: Method for sampling points ("grid" or "random")
            
        Returns:
            Dictionary with synthetic data and visualization info
        """
        if model_name not in self.models:
            return {"error": f"Model '{model_name}' not found"}
        
        model_info = self.models[model_name]
        model = model_info["model"]
        features = model_info["features"]
        target = model_info["target"]
        
        # Check that ranges are provided for all features
        for feature in features:
            if feature not in feature_ranges:
                return {"error": f"Missing range for feature '{feature}'"}
        
        # Generate sample points
        if sampling == "grid" and len(features) <= 3:
            # For 1-3 features, we can use a grid
            feature_values = {}
            for feature, (min_val, max_val, num_points) in feature_ranges.items():
                feature_values[feature] = np.linspace(min_val, max_val, num_points)
            
            # Generate all combinations using meshgrid
            grid_arrays = np.meshgrid(*[feature_values[f] for f in features])
            
            # Reshape to get input matrix
            X_synth = np.vstack([grid.flatten() for grid in grid_arrays]).T
            
        else:
            # Random sampling for higher dimensions or if specified
            X_synth = np.zeros((1000, len(features)))
            for i, feature in enumerate(features):
                min_val, max_val, _ = feature_ranges[feature]
                X_synth[:, i] = np.random.uniform(min_val, max_val, 1000)
        
        # Handle polynomial features if needed
        if model_info["model_type"] == "polynomial":
            poly = PolynomialFeatures(degree=2)
            X_synth_model = poly.fit_transform(X_synth)
            y_synth = model.predict(X_synth_model)
        else:
            # Make predictions
            y_synth = model.predict(X_synth)
        
        # Create a DataFrame with the synthetic data
        synth_data = pd.DataFrame(X_synth, columns=features)
        synth_data[target] = y_synth
        
        # Create a visualization if 1 or 2 features
        plot_data = None
        if len(features) == 1:
            plot_data = self._create_1d_plot(synth_data, features[0], target)
        elif len(features) == 2:
            plot_data = self._create_2d_plot(synth_data, features[0], features[1], target)
        
        # Return results
        return {
            "model_name": model_name,
            "synthetic_data": synth_data.to_dict(orient="records")[:100],  # Limit to 100 points
            "num_points": len(synth_data),
            "features": features,
            "target": target,
            "plot_data": plot_data,
            "message": f"Generated {len(synth_data)} synthetic data points"
        }
    
    def _create_1d_plot(self, data: pd.DataFrame, x_col: str, y_col: str) -> Dict[str, Any]:
        """Create a 1D plot for visualization"""
        plt.figure(figsize=(10, 6))
        plt.scatter(data[x_col], data[y_col], alpha=0.5)
        plt.plot(data[x_col], data[y_col], 'r-', alpha=0.7)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f"Relationship between {x_col} and {y_col}")
        plt.grid(True, alpha=0.3)
        
        # Save to base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        return {
            "type": "1d_plot",
            "x_label": x_col,
            "y_label": y_col,
            "image_data": plot_data
        }
    
    def _create_2d_plot(self, data: pd.DataFrame, x_col: str, y_col: str, z_col: str) -> Dict[str, Any]:
        """Create a 2D plot for visualization"""
        # Create a 3D scatter plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(data[x_col], data[y_col], data[z_col], 
                           c=data[z_col], cmap='viridis', alpha=0.7)
        
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_zlabel(z_col)
        plt.colorbar(scatter, ax=ax, label=z_col)
        plt.title(f"Relationship between {x_col}, {y_col}, and {z_col}")
        
        # Save to base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        return {
            "type": "2d_plot",
            "x_label": x_col,
            "y_label": y_col,
            "z_label": z_col,
            "image_data": plot_data
        }
    
    def extract_key_insights(self, model_name: str) -> Dict[str, Any]:
        """
        Extract key insights from a trained model
        
        Args:
            model_name: Name of the model to analyze
            
        Returns:
            Dictionary with insights about the model
        """
        if model_name not in self.models:
            return {"error": f"Model '{model_name}' not found"}
        
        model_info = self.models[model_name]
        model = model_info["model"]
        features = model_info["features"]
        target = model_info["target"]
        model_type = model_info["model_type"]
        
        insights = {
            "model_name": model_name,
            "target_variable": target,
            "model_type": model_type,
            "model_quality": model_info["metrics"]["r_squared"],
            "key_factors": [],
            "relationships": []
        }
        
        # Extract feature importance based on model type
        if model_type == "linear":
            # For linear models, use coefficients
            for i, feature in enumerate(features):
                coef = model.coef_[i] if hasattr(model.coef_, "__iter__") else model.coef_
                insights["key_factors"].append({
                    "feature": feature,
                    "importance": abs(float(coef)),
                    "effect": "positive" if coef > 0 else "negative"
                })
                
                insights["relationships"].append({
                    "description": f"{feature} has a {'positive' if coef > 0 else 'negative'} relationship with {target}",
                    "magnitude": "strong" if abs(coef) > 0.5 else "moderate" if abs(coef) > 0.2 else "weak",
                    "coefficient": float(coef)
                })
                
        elif model_type == "random_forest":
            # For random forests, use feature importance
            for i, feature in enumerate(features):
                importance = model.feature_importances_[i]
                insights["key_factors"].append({
                    "feature": feature,
                    "importance": float(importance)
                })
        
        # Sort key factors by importance
        insights["key_factors"].sort(key=lambda x: x["importance"], reverse=True)
        
        # Add practical interpretations
        insights["practical_implications"] = []
        
        # Only add if we have relationships
        if len(insights.get("relationships", [])) > 0:
            for rel in insights["relationships"]:
                if rel["magnitude"] == "strong":
                    insights["practical_implications"].append(
                        f"Focus on optimizing {rel['description'].split(' has ')[0]} as it strongly impacts performance."
                    )
        
        return insights
    
    def to_json(self) -> str:
        """Export the model configuration to JSON"""
        export_data = {
            "models": {},
            "data_tables": {},
            "equations": self.equations,
            "variable_relationships": self.variable_relationships
        }
        
        # Add models (without the actual model objects)
        for name, model_info in self.models.items():
            export_data["models"][name] = {
                "model_type": model_info["model_type"],
                "data_table": model_info["data_table"],
                "target": model_info["target"],
                "features": model_info["features"],
                "metrics": model_info["metrics"]
            }
        
        # Add data tables
        for name, df in self.data_tables.items():
            export_data["data_tables"][name] = df.to_dict(orient="records")
        
        return json.dumps(export_data, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'SprintPerformanceModel':
        """Create a model from JSON configuration"""
        data = json.loads(json_str)
        model = cls()
        
        # Load equations
        model.equations = data.get("equations", {})
        
        # Load variable relationships
        model.variable_relationships = data.get("variable_relationships", {})
        
        # Load data tables
        for name, table_data in data.get("data_tables", {}).items():
            df = pd.DataFrame(table_data)
            model.data_tables[name] = df
        
        return model


# Demo data for testing
if __name__ == "__main__":
    # Create model
    model = SprintPerformanceModel()
    
    # Add sample data from Bolt paper
    bolt_data = {
        "stride_length": [2.3, 2.4, 2.5, 2.6, 2.7, 2.8],
        "stride_frequency": [4.5, 4.4, 4.3, 4.2, 4.1, 4.0],
        "velocity": [10.35, 10.56, 10.75, 10.92, 11.07, 11.2]
    }
    
    model.add_data_table("bolt_stride_analysis", bolt_data)
    
    # Train a model
    result = model.train_model(
        "stride_performance", 
        "bolt_stride_analysis",
        "velocity", 
        ["stride_length", "stride_frequency"],
        "linear"
    )
    
    print(result)
    
    # Generate synthetic data
    synth_result = model.generate_synthetic_data(
        "stride_performance",
        {
            "stride_length": (2.0, 3.0, 10),
            "stride_frequency": (3.8, 4.7, 10)
        }
    )
    
    print(f"Generated {synth_result['num_points']} synthetic data points")
    
    # Extract insights
    insights = model.extract_key_insights("stride_performance")
    print(json.dumps(insights, indent=2)) 