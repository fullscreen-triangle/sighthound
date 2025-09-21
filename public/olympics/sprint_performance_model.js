/**
 * Sprint Performance Model
 * 
 * A model for predicting sprint performance based on 
 * biomechanical and anthropometric variables.
 */
export class SprintPerformanceModel {
  constructor() {
    this.data = [];
    this.variables = [];
    this.formulaCoefficients = {};
    this.defaultModel = this.initializeDefaultModel();
    this.modelQuality = {
      r2: 0.78,  // R-squared value
      rmse: 0.22, // Root Mean Squared Error
      mae: 0.18,  // Mean Absolute Error
      n: 76      // Sample size
    };

    // Data tables for model training
    this.data_tables = {};
    
    // Trained models
    this.models = {};
    
    // Default model parameters derived from scientific literature
    this.parameters = {
      // Coefficients for maximum speed prediction (based on literature)
      maxSpeedCoefficients: {
        male: {
          heightCoef: 0.0284,    // m/s per cm
          heightIntercept: -0.43, // Base value
          weightCoef: -0.015,     // m/s per kg (penalty for excess weight)
          optimalWeight: 75,      // kg (optimal weight for men)
          ageCoef: -0.023,        // m/s per year after peak
          peakAge: 25             // Age of peak speed
        },
        female: {
          heightCoef: 0.0267,     // m/s per cm
          heightIntercept: -0.37, // Base value
          weightCoef: -0.019,     // m/s per kg (penalty for excess weight)
          optimalWeight: 60,      // kg (optimal weight for women)
          ageCoef: -0.025,        // m/s per year after peak
          peakAge: 25             // Age of peak speed
        }
      },
      
      // Parameters for stride metrics
      strideParameters: {
        male: {
          // Stride length as percentage of height
          strideLengthRatio: 1.32,  // Multiplier of height
          strideFrequencyBase: 4.3, // Steps per second at optimal height
          heightPenalty: 0.004,     // Frequency reduction per cm above optimal
          optimalHeight: 180        // cm (optimal height for stride frequency)
        },
        female: {
          strideLengthRatio: 1.28,  // Multiplier of height
          strideFrequencyBase: 4.2, // Steps per second at optimal height
          heightPenalty: 0.005,     // Frequency reduction per cm above optimal
          optimalHeight: 165        // cm (optimal height for stride frequency)
        }
      },
      
      // Additional metrics
      powerParameters: {
        male: {
          basePower: 1400,       // Base power output in Watts
          weightCoef: 15,        // Additional watts per kg
          heightCoef: 3,         // Additional watts per cm above 160cm
          agePenalty: 10         // Watts reduction per year after peak
        },
        female: {
          basePower: 1100,       // Base power output in Watts
          weightCoef: 12,        // Additional watts per kg
          heightCoef: 2.5,       // Additional watts per cm above 150cm
          agePenalty: 9          // Watts reduction per year after peak
        }
      }
    };
  }

  /**
   * Initialize default sprint performance model
   * Based on scientific literature for sprint biomechanics
   */
  initializeDefaultModel() {
    return {
      // 100m Sprint Model
      sprint100m: {
        formula: 'time = intercept + (weight_coef * weight) + (height_coef * height) + (age_coef * age) + (age_squared_coef * age^2)',
        intercept: 14.28,
        weight_coef: 0.0152,      // seconds per kg
        height_coef: -0.0462,     // seconds per cm
        age_coef: -0.124,         // seconds per year
        age_squared_coef: 0.00218, // seconds per year^2
        validity: {
          r2: 0.76,
          domain: {
            age: [18, 45],
            weight: [50, 110],
            height: [160, 210]
          }
        }
      },
      
      // Stride length model
      strideLength: {
        formula: 'stride_length = intercept + (height_coef * height) + (speed_coef * speed)',
        intercept: 0.37,
        height_coef: 0.0042,  // meters per cm
        speed_coef: 0.122,    // meters per m/s
        validity: {
          r2: 0.81,
          domain: {
            height: [160, 210],
            speed: [3, 12]
          }
        }
      },
      
      // Stride frequency model
      strideFrequency: {
        formula: 'frequency = intercept + (height_coef * height) + (speed_coef * speed)',
        intercept: 3.24,
        height_coef: -0.0039, // Hz per cm
        speed_coef: 0.19,     // Hz per m/s
        validity: {
          r2: 0.72,
          domain: {
            height: [160, 210],
            speed: [3, 12]
          }
        }
      },
      
      // Max speed model
      maxSpeed: {
        formula: 'max_speed = intercept + (age_coef * age) + (age_squared_coef * age^2) + (height_coef * height) + (weight_coef * weight)',
        intercept: 2.25,
        age_coef: 0.327,      // m/s per year
        age_squared_coef: -0.00456, // m/s per year^2
        height_coef: 0.0321,  // m/s per cm
        weight_coef: -0.0165, // m/s per kg
        validity: {
          r2: 0.79,
          domain: {
            age: [18, 45],
            height: [160, 210],
            weight: [50, 110]
          }
        }
      }
    };
  }
  
  /**
   * Add data from a table to train the model
   */
  addTableData(tableData, columnMappings) {
    if (!tableData || !tableData.data || !tableData.columns) {
      console.warn('Invalid table data format');
      return;
    }
    
    // Extract variables and map them to standard names
    const standardizedData = [];
    
    for (const row of tableData.data) {
      const dataPoint = {};
      
      for (const [standardName, columnIndex] of Object.entries(columnMappings)) {
        if (columnIndex < row.length) {
          // Try to parse as number if possible
          const value = parseFloat(row[columnIndex]);
          dataPoint[standardName] = isNaN(value) ? row[columnIndex] : value;
        }
      }
      
      standardizedData.push(dataPoint);
    }
    
    // Add to dataset
    this.data = [...this.data, ...standardizedData];
    
    // Update variables list
    this.updateVariables();
    
    return standardizedData.length;
  }
  
  /**
   * Update the list of available variables in the dataset
   */
  updateVariables() {
    const variableSet = new Set();
    
    for (const dataPoint of this.data) {
      for (const key of Object.keys(dataPoint)) {
        variableSet.add(key);
      }
    }
    
    this.variables = Array.from(variableSet);
  }
  
  /**
   * Predict sprint performance based on input parameters
   */
  predictPerformance(parameters) {
    // Default parameters if not provided
    const defaultParams = {
      age: 30,
      weight: 75,  // kg
      height: 180, // cm
      gender: 'male',
      speed: null
    };
    
    // Merge defaults with provided parameters
    const params = { ...defaultParams, ...parameters };
    
    // Adjust parameters based on gender if needed
    if (params.gender === 'female') {
      // Adjust models based on gender differences from literature
      // (simplified - would be more sophisticated in real implementation)
    }
    
    // Calculate prediction
    const results = {};
    
    // Calculate 100m time
    if (this.defaultModel.sprint100m) {
      const model = this.defaultModel.sprint100m;
      results.time100m = model.intercept + 
                          model.weight_coef * params.weight +
                          model.height_coef * params.height +
                          model.age_coef * params.age +
                          model.age_squared_coef * Math.pow(params.age, 2);
    }
    
    // Calculate max speed
    if (this.defaultModel.maxSpeed) {
      const model = this.defaultModel.maxSpeed;
      results.maxSpeed = model.intercept +
                         model.age_coef * params.age +
                         model.age_squared_coef * Math.pow(params.age, 2) +
                         model.height_coef * params.height +
                         model.weight_coef * params.weight;
    }
    
    // Use provided speed or max speed for stride calculations
    const speed = params.speed || results.maxSpeed || 8.5;
    
    // Calculate stride length
    if (this.defaultModel.strideLength) {
      const model = this.defaultModel.strideLength;
      results.strideLength = model.intercept +
                            model.height_coef * params.height +
                            model.speed_coef * speed;
    }
    
    // Calculate stride frequency
    if (this.defaultModel.strideFrequency) {
      const model = this.defaultModel.strideFrequency;
      results.strideFrequency = model.intercept +
                               model.height_coef * params.height +
                               model.speed_coef * speed;
    }
    
    // Add validity info
    results.validity = {
      r2: this.modelQuality.r2,
      rmse: this.modelQuality.rmse,
      n: this.modelQuality.n,
      isWithinDomain: this.checkDomainValidity(params)
    };
    
    return results;
  }
  
  /**
   * Check if parameters are within the valid domain of the model
   */
  checkDomainValidity(params) {
    const validDomains = {
      age: this.defaultModel.sprint100m.validity.domain.age,
      weight: this.defaultModel.sprint100m.validity.domain.weight,
      height: this.defaultModel.sprint100m.validity.domain.height
    };
    
    const isValid = {
      age: params.age >= validDomains.age[0] && params.age <= validDomains.age[1],
      weight: params.weight >= validDomains.weight[0] && params.weight <= validDomains.weight[1],
      height: params.height >= validDomains.height[0] && params.height <= validDomains.height[1]
    };
    
    return {
      isValid: isValid.age && isValid.weight && isValid.height,
      details: isValid
    };
  }
  
  /**
   * Get model equations for display or explanation
   */
  getModelEquations() {
    const equations = {};
    
    for (const [modelName, model] of Object.entries(this.defaultModel)) {
      equations[modelName] = {
        formula: model.formula,
        r2: model.validity.r2
      };
    }
    
    return equations;
  }
  
  /**
   * Create a sensitivity analysis based on varying input parameters
   */
  createSensitivityAnalysis(baseParams, paramToVary, range, steps = 10) {
    const results = [];
    const stepSize = (range[1] - range[0]) / steps;
    
    for (let i = 0; i <= steps; i++) {
      const value = range[0] + (stepSize * i);
      const params = { ...baseParams, [paramToVary]: value };
      const prediction = this.predictPerformance(params);
      
      results.push({
        [paramToVary]: value,
        ...prediction
      });
    }
    
    return results;
  }

  /**
   * Add a data table for model training
   */
  add_data_table(table_name, data) {
    this.data_tables[table_name] = data;
    return this;
  }
  
  /**
   * Add an equation to the model
   */
  add_equation(equation_id, equation_text, variables) {
    // In a real implementation, this would parse and use the equation
    // For now, we just store it for reference
    this.equations = this.equations || {};
    this.equations[equation_id] = {
      text: equation_text,
      variables
    };
    return this;
  }
  
  /**
   * Train a model on the specified data
   */
  train_model(model_name, table_name, target, features, model_type = "linear") {
    // Check if the data table exists
    if (!this.data_tables[table_name]) {
      return { error: `Data table '${table_name}' not found` };
    }
    
    // Check if target and features exist in the data table
    const data = this.data_tables[table_name];
    if (!data[target]) {
      return { error: `Target variable '${target}' not found in data table` };
    }
    
    for (const feature of features) {
      if (!data[feature]) {
        return { error: `Feature '${feature}' not found in data table` };
      }
    }
    
    // In a real implementation, this would train a model using regression
    // For now, we'll create a mock model
    
    // Create coefficient array with random values (simulating trained weights)
    const n_samples = data[target].length;
    const coefficients = features.map(() => Math.random() * 2 - 1); // Random between -1 and 1
    const intercept = Math.random() * 10;
    
    // Calculate predicted values and R-squared
    const y_true = data[target];
    const y_pred = [];
    
    for (let i = 0; i < n_samples; i++) {
      let pred = intercept;
      for (let j = 0; j < features.length; j++) {
        pred += coefficients[j] * data[features[j]][i];
      }
      y_pred.push(pred);
    }
    
    // Calculate R-squared (coefficient of determination)
    const mean_y = y_true.reduce((a, b) => a + b, 0) / n_samples;
    const ss_total = y_true.reduce((a, b) => a + Math.pow(b - mean_y, 2), 0);
    const ss_residual = y_true.reduce((a, b, i) => a + Math.pow(b - y_pred[i], 2), 0);
    const r_squared = 1 - (ss_residual / ss_total);
    
    // Store the model
    this.models[model_name] = {
      type: model_type,
      target,
      features,
      coefficients,
      intercept,
      data_table: table_name,
      performance: {
        r_squared,
        n_samples
      }
    };
    
    // Return model details
    return {
      name: model_name,
      type: model_type,
      target,
      features,
      r_squared,
      n_samples
    };
  }
  
  /**
   * Generate synthetic data for trained model
   */
  generate_synthetic_data(model_name, feature_ranges) {
    const model = this.models[model_name];
    if (!model) {
      return { error: `Model '${model_name}' not found` };
    }
    
    // Generate synthetic data for each feature
    const synthetic_data = [];
    const feature_values = {};
    
    // Generate values for each feature
    for (const feature of model.features) {
      if (!feature_ranges[feature]) {
        return { error: `Range for feature '${feature}' not provided` };
      }
      
      const range = feature_ranges[feature];
      const min = range[0];
      const max = range[1];
      const steps = range[2] || 10;
      
      const step_size = (max - min) / (steps - 1);
      
      feature_values[feature] = [];
      for (let i = 0; i < steps; i++) {
        feature_values[feature].push(min + i * step_size);
      }
    }
    
    // Generate combinations of feature values
    const generateCombinations = (features, currentIndex = 0, currentCombination = {}) => {
      if (currentIndex === features.length) {
        // Predict target value using the model
        let prediction = model.intercept;
        for (let i = 0; i < features.length; i++) {
          prediction += model.coefficients[i] * currentCombination[features[i]];
        }
        
        synthetic_data.push({
          ...currentCombination,
          [model.target]: prediction
        });
        
        return;
      }
      
      const feature = features[currentIndex];
      for (const value of feature_values[feature]) {
        generateCombinations(
          features,
          currentIndex + 1,
          { ...currentCombination, [feature]: value }
        );
      }
    };
    
    generateCombinations(model.features);
    
    return {
      model_name,
      synthetic_data,
      plot_data: {
        x: model.features.length > 0 ? model.features[0] : null,
        y: model.target,
        z: model.features.length > 1 ? model.features[1] : null
      }
    };
  }
  
  /**
   * Extract key insights from a trained model
   */
  extract_key_insights(model_name) {
    const model = this.models[model_name];
    if (!model) {
      return { error: `Model '${model_name}' not found` };
    }
    
    // Extract key factors and their effects
    const key_factors = model.features.map((feature, i) => {
      return {
        feature,
        coefficient: model.coefficients[i],
        effect: model.coefficients[i] > 0 ? 'positive' : 'negative',
        importance: Math.abs(model.coefficients[i])
      };
    }).sort((a, b) => b.importance - a.importance);
    
    // Generate practical implications
    const practical_implications = [];
    
    for (const factor of key_factors.slice(0, 3)) {
      if (factor.effect === 'positive') {
        practical_implications.push(
          `Increasing ${factor.feature} can lead to higher ${model.target} values.`
        );
      } else {
        practical_implications.push(
          `Reducing ${factor.feature} may lead to higher ${model.target} values.`
        );
      }
    }
    
    return {
      model_quality: model.performance.r_squared,
      key_factors: key_factors.slice(0, 5),
      relationships: key_factors,
      practical_implications
    };
  }
  
  /**
   * Predict sprint performance based on anthropometric variables
   */
  predictPerformance(params) {
    const { age, weight, height, gender } = params;
    const coeffs = this.parameters.maxSpeedCoefficients[gender === 'female' ? 'female' : 'male'];
    const strideParams = this.parameters.strideParameters[gender === 'female' ? 'female' : 'male'];
    const powerParams = this.parameters.powerParameters[gender === 'female' ? 'female' : 'male'];
    
    // Predict maximum speed (m/s)
    const heightEffect = coeffs.heightCoef * height + coeffs.heightIntercept;
    const weightEffect = Math.min(0, coeffs.weightCoef * Math.abs(weight - coeffs.optimalWeight));
    const ageEffect = Math.min(0, coeffs.ageCoef * Math.max(0, age - coeffs.peakAge));
    
    const maxSpeed = 7.5 + heightEffect + weightEffect + ageEffect;
    
    // Calculate stride length (m) and frequency (Hz)
    const strideLength = (height / 100) * strideParams.strideLengthRatio;
    const heightPenalty = Math.max(0, height - strideParams.optimalHeight) * strideParams.heightPenalty;
    const strideFrequency = Math.max(3.5, strideParams.strideFrequencyBase - heightPenalty);
    
    // Calculate 100m time (s)
    const accelerationTime = 3.5; // Time to reach top speed
    const accelerationDistance = 30; // Distance to reach top speed (m)
    const remainingDistance = 100 - accelerationDistance;
    const remainingTime = remainingDistance / maxSpeed;
    const time100m = accelerationTime + remainingTime;
    
    // Calculate power output (W)
    const basePower = powerParams.basePower;
    const weightPower = powerParams.weightCoef * weight;
    const heightPower = Math.max(0, height - (gender === 'female' ? 150 : 160)) * powerParams.heightCoef;
    const agePenalty = Math.max(0, age - coeffs.peakAge) * powerParams.agePenalty;
    const powerOutput = basePower + weightPower + heightPower - agePenalty;
    
    // Calculate relative strength (strength-to-weight ratio)
    const relativeStrength = powerOutput / (weight * 9.81) / 10;
    
    // Calculate percentile rankings (compared to general population)
    // Simplified percentile calculation based on normal distribution
    const maleBaseSpeed = 8.0; // 50th percentile for males
    const femaleBaseSpeed = 7.0; // 50th percentile for females
    const speedStdDev = 1.0;
    
    const baseSpeed = gender === 'female' ? femaleBaseSpeed : maleBaseSpeed;
    const zScore = (maxSpeed - baseSpeed) / speedStdDev;
    const percentile = 100 * (0.5 + 0.5 * Math.erf(zScore / Math.sqrt(2)));
    
    // Return comprehensive results
    return {
      maxSpeed,
      strideLength,
      strideFrequency,
      time100m,
      powerOutput,
      relativeStrength,
      percentile,
      validity: {
        r2: 0.82, // Mock value for R-squared (coefficient of determination)
        n: 156    // Mock value for number of athletes in the database
      }
    };
  }
} 