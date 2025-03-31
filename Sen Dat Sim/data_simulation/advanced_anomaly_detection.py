import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import time
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_curve, auc
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import BatchNormalization

class AdvancedAnomalyDetection:
    """Advanced anomaly detection models for predictive maintenance"""
    
    def __init__(self, model_dir='models'):
        """ Initialize the anomaly detection system """
        self.model_dir = model_dir
        self.models = {}
        self.scalers = {}
        self.threshold = 0.5
        self.sequence_length = 60
        
        # Create model directory if it doesn't exist
        import os
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
    def prepare_sequence_data(self, X, y=None, sequence_length=None):
        """
        Prepare data for sequence-based models like LSTM
        
        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            Features data
        y : pandas.Series or numpy.ndarray, optional
            Target labels
        sequence_length : int, optional
            Length of sequences to create
            
        Returns:
        --------
        tuple
            (X_sequences, y_sequences) if y is provided, else just X_sequences
        """
        if sequence_length is None:
            sequence_length = self.sequence_length
            
        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        if y is not None and isinstance(y, pd.Series):
            y = y.values
            
        # Create sequences
        X_sequences = []
        for i in range(len(X) - sequence_length + 1):
            X_sequences.append(X[i:i+sequence_length])
        X_sequences = np.array(X_sequences)
        
        # If y is provided, create corresponding sequences
        if y is not None:
            y_sequences = []
            for i in range(len(y) - sequence_length + 1):
                # Use the label at the end of the sequence
                y_sequences.append(y[i+sequence_length-1])
            y_sequences = np.array(y_sequences)
            
            return X_sequences, y_sequences
        else:
            return X_sequences
    
    def train_isolation_forest(self, X_train, contamination=0.05):
        """
        Train an Isolation Forest model for unsupervised anomaly detection
        
        Parameters:
        -----------
        X_train : pandas.DataFrame
            Training data features
        contamination : float
            Expected proportion of anomalies in the data
            
        Returns:
        --------
        self : for method chaining
        """
        print("Training Isolation Forest model...")
        
        # Data Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Model Training
        model = IsolationForest(
            n_estimators=100,
            max_samples='auto', 
            contamination=contamination,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train_scaled)
        
        # Save the model and scaler
        self.models['isolation_forest'] = model
        self.scalers['isolation_forest'] = scaler
        
        return self
    
    def train_one_class_svm(self, X_train, nu=0.05):
        """
        Train a One-Class SVM model for unsupervised anomaly detection
        
        Parameters:
        -----------
        X_train : pandas.DataFrame
            Training data features
        nu : float
            An upper bound on the fraction of training errors and a lower bound of the 
            fraction of support vectors
            
        Returns:
        --------
        self : for method chaining
        """
        print("Training One-Class SVM model...")
        
        # Data Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Model Training
        model = OneClassSVM(
            kernel='rbf',
            gamma='scale',
            nu=nu
        )
        model.fit(X_train_scaled)
        
        # Save the model and scaler
        self.models['one_class_svm'] = model
        self.scalers['one_class_svm'] = scaler
        
        return self
    
    def train_autoencoder(self, X_train, validation_split=0.1, epochs=50, batch_size=32):
        """
        Train an Autoencoder model for unsupervised anomaly detection
        
        Parameters:
        -----------
        X_train : pandas.DataFrame
            Training data features
        validation_split : float
            Fraction of training data to use for validation
        epochs : int
            Number of training epochs
        batch_size : int
            Training batch size
            
        Returns:
        --------
        self : for method chaining
        """
        print("Training Autoencoder model...")
        
        # Data Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Define the model architecture
        input_dim = X_train.shape[1]
        encoding_dim = min(32, input_dim // 2)  # Reduce dimension for encoding
        

        input_layer = Input(shape=(input_dim,))

        encoded = Dense(encoding_dim * 2, activation='relu')(input_layer)
        encoded = Dense(encoding_dim, activation='relu')(encoded)
        
        decoded = Dense(encoding_dim * 2, activation='relu')(encoded)
        decoded = Dense(input_dim, activation='sigmoid')(decoded)

        autoencoder = Model(input_layer, decoded)
        
        autoencoder.compile(optimizer='adam', loss='mean_squared_error')
        
        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        # Model Training
        history = autoencoder.fit(
            X_train_scaled, 
            X_train_scaled,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Save the model and scaler
        self.models['autoencoder'] = autoencoder
        self.scalers['autoencoder'] = scaler
        
        # Calculate reconstruction error threshold
        reconstructions = autoencoder.predict(X_train_scaled)
        reconstruction_errors = np.mean(np.square(X_train_scaled - reconstructions), axis=1)
        
        # Set threshold as mean + 2*std of reconstruction errors
        self.thresholds = {}
        self.thresholds['autoencoder'] = np.mean(reconstruction_errors) + 2 * np.std(reconstruction_errors)
        
        return self
    
    def train_improved_autoencoder(self, X_train, validation_split=0.2, epochs=200, batch_size=32, patience=20):
        """Train an improved autoencoder with better architecture and training parameters"""
        print("Training enhanced autoencoder model...")
        
        # Data Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Split data into train and validation
        X_train_split, X_val = train_test_split(X_train_scaled, test_size=validation_split, random_state=42)
        
        # Define improved architecture
        input_dim = X_train.shape[1]
        encoding_dim = min(64, input_dim)  # Larger encoding dimension
        
        input_layer = Input(shape=(input_dim,))
        
        # Encoder with batch normalization
        encoded = Dense(encoding_dim * 2, activation='relu')(input_layer)
        encoded = BatchNormalization()(encoded)
        encoded = Dense(encoding_dim, activation='relu')(encoded)
        encoded = BatchNormalization()(encoded)
        
        # Bottleneck
        bottleneck = Dense(int(encoding_dim/2), activation='relu')(encoded)
        
        # Decoder with batch normalization
        decoded = Dense(encoding_dim, activation='relu')(bottleneck)
        decoded = BatchNormalization()(decoded)
        decoded = Dense(encoding_dim * 2, activation='relu')(decoded)
        decoded = BatchNormalization()(decoded)
        decoded = Dense(input_dim, activation='sigmoid')(decoded)
        
        # Autoencoder model
        autoencoder = Model(input_layer, decoded)
        
        # Learning rate schedule
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=10000,
            decay_rate=0.9)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        
        # Compile with appropriate loss function
        autoencoder.compile(optimizer=optimizer, loss='mean_squared_error')
        
        # Early stopping with higher patience
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
        
        # Model checkpoint to save best model
        checkpoint = ModelCheckpoint(
            f"{self.model_dir}/best_autoencoder.h5",
            save_best_only=True,
            monitor='val_loss'
        )
        
        # Model Training with progress tracking
        history = autoencoder.fit(
            X_train_split, 
            X_train_split,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_data=(X_val, X_val),
            callbacks=[early_stopping, checkpoint],
            verbose=1
        )
        
        # Load the best model
        autoencoder = load_model(f"{self.model_dir}/best_autoencoder.h5")
        
        # Save the model and scaler
        self.models['enhanced_autoencoder'] = autoencoder
        self.scalers['enhanced_autoencoder'] = scaler
        
        # Calibrate thresholds more carefully
        reconstructions = autoencoder.predict(X_train_scaled)
        reconstruction_errors = np.mean(np.square(X_train_scaled - reconstructions), axis=1)
        
        # Use percentile-based threshold instead of mean+std
        if not hasattr(self, 'thresholds'):
            self.thresholds = {}
        
        # Set threshold at 95th percentile of reconstruction errors
        self.thresholds['enhanced_autoencoder'] = np.percentile(reconstruction_errors, 95)
        
        # Plot training history
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Autoencoder Training History')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.model_dir}/autoencoder_training_history.png")
        
        return self, history
    
    def train_lstm_autoencoder(self, X_train, validation_split=0.1, epochs=50, batch_size=32):
        """
        Train an LSTM Autoencoder model for sequence-based anomaly detection
        
        Parameters:
        -----------
        X_train : pandas.DataFrame
            Training data features
        validation_split : float
            Fraction of training data to use for validation
        epochs : int
            Number of training epochs
        batch_size : int
            Training batch size
            
        Returns:
        --------
        self : for method chaining
        """
        print("Training LSTM Autoencoder model...")
        
        # Data Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Prepare sequences
        X_train_seq = self.prepare_sequence_data(X_train_scaled)
        
        # Define model architecture
        timesteps = X_train_seq.shape[1]
        input_dim = X_train_seq.shape[2]
        latent_dim = 32
        
        model = Sequential([
            # Encoder
            LSTM(latent_dim * 2, activation='tanh', return_sequences=True, input_shape=(timesteps, input_dim)),
            Dropout(0.2),
            LSTM(latent_dim, activation='tanh', return_sequences=False),
            
            # Decoder
            RepeatVector(timesteps),
            LSTM(latent_dim, activation='tanh', return_sequences=True),
            Dropout(0.2),
            LSTM(latent_dim * 2, activation='tanh', return_sequences=True),
            TimeDistributed(Dense(input_dim))
        ])
        
        # Compile the model
        model.compile(optimizer='adam', loss='mse')
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        model_checkpoint = ModelCheckpoint(
            f"{self.model_dir}/lstm_autoencoder.h5",
            save_best_only=True,
            monitor='val_loss'
        )
        
        # Model Training
        history = model.fit(
            X_train_seq, 
            X_train_seq,  # Target is the same as input for autoencoder
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping, model_checkpoint],
            verbose=1
        )
        
        # Save the model and scaler
        self.models['lstm_autoencoder'] = model
        self.scalers['lstm_autoencoder'] = scaler
        
        # Calculate reconstruction error threshold
        reconstructions = model.predict(X_train_seq)
        # Mean squared error per sample and timestep
        mse = np.mean(np.square(X_train_seq - reconstructions), axis=(1, 2))
        
        # Set threshold as mean + 2*std of reconstruction errors
        if not hasattr(self, 'thresholds'):
            self.thresholds = {}
        self.thresholds['lstm_autoencoder'] = np.mean(mse) + 2 * np.std(mse)
        
        return self
    
    def train_supervised_rf(self, X_train, y_train):
        """
        Train a supervised Random Forest model for anomaly detection
        
        Parameters:
        -----------
        X_train : pandas.DataFrame
            Training data features
        y_train : pandas.Series
            Target labels (1 for anomaly, 0 for normal)
            
        Returns:
        --------
        self : for method chaining
        """
        print("Training supervised Random Forest model...")
        
        # Create a pipeline with scaling and model
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            ))
        ])
        
        # Model Training
        pipeline.fit(X_train, y_train)
        
        # Save the model
        self.models['random_forest'] = pipeline
        
        return self
    
    def train_supervised_xgboost(self, X_train, y_train):
        """
        Train a supervised XGBoost model for anomaly detection
        
        Parameters:
        -----------
        X_train : pandas.DataFrame
            Training data features
        y_train : pandas.Series
            Target labels (1 for anomaly, 0 for normal)
            
        Returns:
        --------
        self : for method chaining
        """
        print("Training supervised XGBoost model...")
        
        # Create a pipeline with scaling and model
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss',
                scale_pos_weight=sum(y_train==0)/sum(y_train==1)  # Balance classes
            ))
        ])
        
        # Model Training
        pipeline.fit(X_train, y_train)
        
        # Save the model
        self.models['xgboost'] = pipeline
        
        return self
    
    def train_lstm_supervised(self, X_train, y_train, validation_split=0.1, epochs=50, batch_size=32):
        """
        Train a supervised LSTM model for sequence-based anomaly detection
        
        Parameters:
        -----------
        X_train : pandas.DataFrame
            Training data features
        y_train : pandas.Series
            Target labels (1 for anomaly, 0 for normal)
        validation_split : float
            Fraction of training data to use for validation
        epochs : int
            Number of training epochs
        batch_size : int
            Training batch size
            
        Returns:
        --------
        self : for method chaining
        """
        print("Training supervised LSTM model...")
        
        # Data Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Prepare sequences
        X_train_seq, y_train_seq = self.prepare_sequence_data(X_train_scaled, y_train)
        
        # Define model architecture
        timesteps = X_train_seq.shape[1]
        input_dim = X_train_seq.shape[2]
        
        # Build the model
        model = Sequential([
            LSTM(64, activation='tanh', return_sequences=True, input_shape=(timesteps, input_dim)),
            Dropout(0.2),
            LSTM(32, activation='tanh'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        model_checkpoint = ModelCheckpoint(
            f"{self.model_dir}/lstm_supervised.h5",
            save_best_only=True,
            monitor='val_loss'
        )
        
        # Model Training
        history = model.fit(
            X_train_seq, 
            y_train_seq,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping, model_checkpoint],
            verbose=1
        )
        
        # Save the model and scaler
        self.models['lstm_supervised'] = model
        self.scalers['lstm_supervised'] = scaler
        
        return self
    
    def tune_model_hyperparameters(self, X_train, y_train, model_type='random_forest'):
        """
        Tune hyperparameters for a specified model type using GridSearchCV
        
        Parameters:
        -----------
        X_train : pandas.DataFrame
            Training data features
        y_train : pandas.Series
            Target labels (1 for anomaly, 0 for normal)
        model_type : str
            Type of model to tune ('random_forest', 'xgboost', 'isolation_forest', etc.)
            
        Returns:
        --------
        dict
            Best parameters found
        """
        print(f"Tuning hyperparameters for {model_type} model...")
        
        # Define parameters
        param_grids = {
            'random_forest': {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__max_depth': [None, 10, 20],
                'classifier__min_samples_split': [2, 5, 10]
            },
            'xgboost': {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__learning_rate': [0.01, 0.1, 0.2],
                'classifier__max_depth': [3, 5, 7]
            },
            'isolation_forest': {
                'n_estimators': [50, 100, 200],
                'max_samples': [0.5, 0.8, 'auto'],
                'contamination': [0.01, 0.05, 0.1]
            },
            'one_class_svm': {
                'nu': [0.01, 0.05, 0.1],
                'gamma': ['scale', 'auto', 0.1]
            }
        }
        
        # Get parameters
        if model_type not in param_grids:
            raise ValueError(f"No parameter grid defined for model type '{model_type}'")
        
        param_grid = param_grids[model_type]
        
        # Create base model based on model type
        if model_type == 'random_forest':
            base_model = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1))
            ])
        elif model_type == 'xgboost':
            base_model = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'))
            ])
        elif model_type == 'isolation_forest':
            base_model = IsolationForest(random_state=42, n_jobs=-1)
        elif model_type == 'one_class_svm':
            base_model = OneClassSVM(kernel='rbf')
            
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=5,
            scoring='f1' if model_type in ['random_forest', 'xgboost'] else 'neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit GridSearchCV
        if model_type in ['isolation_forest', 'one_class_svm']:
            # For unsupervised models
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            grid_search.fit(X_train_scaled)
        else:
            # For supervised models
            grid_search.fit(X_train, y_train)
            
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best score: {grid_search.best_score_}")
        
        # Save the best model
        self.models[f'{model_type}_tuned'] = grid_search.best_estimator_
        
        if model_type in ['isolation_forest', 'one_class_svm']:
            self.scalers[f'{model_type}_tuned'] = scaler
            
        return grid_search.best_params_
    
    def predict_anomalies(self, X, model_name=None):
        """
        Predict anomalies using a trained model
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Test data features
        model_name : str or None
            Name of the model to use. If None, uses all available models.
            
        Returns:
        --------
        dict
            Dictionary of predictions for each model
        """
        if not self.models:
            raise ValueError("No trained models available. Train models first.")
            
        # Ensure X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Handle NaN values
        if X.isna().sum().sum() > 0:
            print(f"Warning: Input data contains {X.isna().sum().sum()} NaN values. Replacing with zeros.")
            X = X.fillna(0)
            
        # Determine which models to use
        if model_name is not None:
            if model_name not in self.models:
                raise ValueError(f"Model '{model_name}' not found. Available models: {list(self.models.keys())}")
            model_names = [model_name]
        else:
            model_names = list(self.models.keys())
        
        # Dictionary to store results
        results = {}
        
        # Make predictions with each model
        for name in model_names:
            try:
                model = self.models[name]
                
                # Prepare the data
                if name in self.scalers:
                    scaler = self.scalers[name]
                    
                    # Feature alignment - ensure columns match training data
                    if hasattr(scaler, 'feature_names_in_'):
                        expected_columns = scaler.feature_names_in_
                        
                        # Get current columns
                        current_columns = set(X.columns)
                        
                        # Find missing columns
                        missing_columns = [col for col in expected_columns if col not in current_columns]
                        
                        # Create a copy of X to modify
                        X_aligned = X.copy()
                        
                        # Add missing columns with zeros
                        for col in missing_columns:
                            X_aligned[col] = 0
                        
                        # Check if we need to mention the feature difference
                        if missing_columns:
                            print(f"Adding {len(missing_columns)} missing features for model {name}")
                            if len(missing_columns) < 10:  # Only show if the list is short
                                print(f"Missing features: {missing_columns}")
                        
                        # Reorder columns to match what the model expects
                        X_aligned = X_aligned[expected_columns]
                        
                        # Transform the aligned data
                        X_scaled = scaler.transform(X_aligned)
                    else:
                        # No feature names available, just scale what we have
                        X_scaled = scaler.transform(X)
                else:
                    X_scaled = X
                    
                # Make predictions based on model type
                if name in ['isolation_forest', 'one_class_svm']:
                    # These models return -1 for anomalies and 1 for normal points
                    raw_predictions = model.predict(X_scaled)
                    # Convert to 1 for anomaly, 0 for normal
                    predictions = np.where(raw_predictions == -1, 1, 0)
                    
                    # Add anomaly scores if available
                    if hasattr(model, 'decision_function'):
                        scores = -model.decision_function(X_scaled)  # Negative because higher scores = more anomalous
                        results[f'{name}_score'] = scores
                        
                elif 'autoencoder' in name:
                    # For autoencoder models, calculate reconstruction error
                    if 'lstm' in name:
                        # Prepare sequences for LSTM autoencoder
                        X_seq = self.prepare_sequence_data(X_scaled)
                        reconstructions = model.predict(X_seq)
                        # Reconstruction error
                        mse = np.mean(np.square(X_seq - reconstructions), axis=(1, 2))
                        # Apply threshold
                        threshold = self.thresholds.get(name, np.mean(mse) + 3 * np.std(mse))
                        predictions = np.where(mse > threshold, 1, 0)
                        results[f'{name}_score'] = mse
                    else:
                        # Standard autoencoder
                        reconstructions = model.predict(X_scaled)
                        # Reconstruction error
                        mse = np.mean(np.square(X_scaled - reconstructions), axis=1)
                        # Apply threshold
                        threshold = self.thresholds.get(name, np.mean(mse) + 2 * np.std(mse))
                        predictions = np.where(mse > threshold, 1, 0)
                        results[f'{name}_score'] = mse
                        
                elif 'lstm_supervised' in name:
                    # Prepare sequences for supervised LSTM
                    X_seq = self.prepare_sequence_data(X_scaled)
                    # Get probabilities
                    probabilities = model.predict(X_seq).flatten()
                    # Apply threshold
                    predictions = np.where(probabilities > 0.5, 1, 0)
                    results[f'{name}_prob'] = probabilities
                    
                else:
                    # For sklearn pipelines (Random Forest, XGBoost)
                    try:
                        # Check if it's a pipeline with a 'classifier' step
                        if hasattr(model, 'named_steps') and 'classifier' in model.named_steps:
                            # Get the feature names from the scaler in the pipeline
                            if hasattr(model.named_steps['scaler'], 'feature_names_in_'):
                                expected_columns = model.named_steps['scaler'].feature_names_in_
                                
                                # Get current columns
                                current_columns = set(X.columns)
                                
                                # Find missing columns
                                missing_columns = [col for col in expected_columns if col not in current_columns]
                                
                                # Create a copy of X to modify
                                X_for_pipeline = X.copy()
                                
                                # Add missing columns with zeros
                                for col in missing_columns:
                                    X_for_pipeline[col] = 0
                                
                                # Check if we need to mention the feature difference
                                if missing_columns:
                                    print(f"Adding {len(missing_columns)} missing features for pipeline model {name}")
                                    if len(missing_columns) < 10:  # Only show if the list is short
                                        print(f"Missing features: {missing_columns}")
                                
                                # Reorder columns to match what the model expects
                                X_for_pipeline = X_for_pipeline[expected_columns]
                                
                                # Use the aligned data with the pipeline
                                probabilities = model.predict_proba(X_for_pipeline)[:, 1]
                                results[f'{name}_prob'] = probabilities
                                predictions = model.predict(X_for_pipeline)
                            else:
                                # No feature names available, try direct prediction
                                probabilities = model.predict_proba(X)[:, 1]
                                results[f'{name}_prob'] = probabilities
                                predictions = model.predict(X)
                        else:
                            # Not a pipeline with expected structure
                            probabilities = model.predict_proba(X_scaled)[:, 1]
                            results[f'{name}_prob'] = probabilities
                            predictions = model.predict(X_scaled)
                    except (IndexError, AttributeError, ValueError) as e:
                        print(f"Error with probabilistic prediction for {name}: {e}")
                        # Fall back to just predict
                        try:
                            predictions = model.predict(X_scaled)
                        except Exception as e2:
                            print(f"Could not make predictions with {name}: {e2}")
                            # Skip this model
                            continue
                        
                # Store predictions
                results[name] = predictions
                
            except Exception as e:
                print(f"Error predicting with model {name}: {e}")
                # Skip this model and continue with others
                continue
        
        return results

    def predict_with_confidence(self, X, model_name=None, calibration_window=None):
        """
        Predict anomalies with confidence scores
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Test data features
        model_name : str or None
            Name of the model to use
        calibration_window : pandas.DataFrame or None
            Recent data for threshold calibration
            
        Returns:
        --------
        dict
            Dictionary with predictions, confidence scores, and thresholds
        """
        # First perform regular prediction
        results = self.predict_anomalies(X, model_name)
        
        # If calibration window provided, recalibrate thresholds
        if calibration_window is not None:
            for name in results.keys():
                if name in self.models and not name.endswith('_prob') and not name.endswith('_score'):
                    self.calculate_dynamic_threshold(calibration_window, name)
        
        # Add confidence scores
        confidence_scores = {}
        
        for name in results.keys():
            if name.endswith('_score'):
                model_name = name.replace('_score', '')
                if model_name in self.thresholds:
                    threshold = self.thresholds[model_name]
                    scores = results[name]
                    # Normalize to 0-1 range
                    confidence = (scores - threshold) / (threshold + 1e-10)
                    confidence = np.clip(1 / (1 + np.exp(-confidence * 5)), 0, 1)  # Sigmoid scaling
                    confidence_scores[model_name] = confidence
        
        # Add to results
        results['confidence_scores'] = confidence_scores
        results['thresholds'] = {k: v for k, v in self.thresholds.items()}
        
        return results
    
    def evaluate_model(self, X_test, y_test, model_name):
        """
        Evaluate a trained model on test data
        
        Parameters:
        -----------
        X_test : pandas.DataFrame
            Test data features
        y_test : pandas.Series
            True labels (1 for anomaly, 0 for normal)
        model_name : str
            Name of the model to evaluate
            
        Returns:
        --------
        dict
            Evaluation metrics
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(self.models.keys())}")
            
        # Get predictions
        predictions = self.predict_anomalies(X_test, model_name)
        y_pred = predictions[model_name]
        
        # Calculate metrics
        metrics = {}
        metrics['classification_report'] = classification_report(y_test, y_pred)
        metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred)
        
        # Calculate AUC-ROC and Precision-Recall curves if scores/probabilities are available
        score_key = f'{model_name}_prob' if f'{model_name}_prob' in predictions else f'{model_name}_score'
        if score_key in predictions:
            scores = predictions[score_key]
            
            # ROC curve and AUC
            fpr, tpr, roc_thresholds = roc_curve(y_test, scores)
            metrics['roc_auc'] = auc(fpr, tpr)
            metrics['roc_curve'] = (fpr, tpr, roc_thresholds)
            
            # Precision-Recall curve
            precision, recall, pr_thresholds = precision_recall_curve(y_test, scores)
            metrics['pr_curve'] = (precision, recall, pr_thresholds)
            
            # F1 score at different thresholds
            f1_scores = []
            for threshold in np.linspace(0, 1, 100):
                y_pred_at_threshold = np.where(scores > threshold, 1, 0)
                f1 = self._calculate_f1(y_test, y_pred_at_threshold)
                f1_scores.append((threshold, f1))
            metrics['f1_thresholds'] = f1_scores
            
            # Find optimal threshold
            optimal_threshold, optimal_f1 = max(f1_scores, key=lambda x: x[1])
            metrics['optimal_threshold'] = optimal_threshold
            metrics['optimal_f1'] = optimal_f1
        
        print(f"Model: {model_name}")
        print(metrics['classification_report'])
        
        # Print AUC-ROC if available
        if 'roc_auc' in metrics:
            print(f"AUC-ROC: {metrics['roc_auc']:.4f}")
            
        # Print optimal threshold if available
        if 'optimal_threshold' in metrics:
            print(f"Optimal threshold: {metrics['optimal_threshold']:.4f} (F1: {metrics['optimal_f1']:.4f})")
            
        return metrics
    
    def _calculate_f1(self, y_true, y_pred):
        """Calculate F1 score manually to handle edge cases"""
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return f1
    
    def plot_evaluation_curves(self, metrics, model_name):
        """
        Plot ROC and Precision-Recall curves for a model
        
        Parameters:
        -----------
        metrics : dict
            Metrics dictionary from evaluate_model
        model_name : str
            Name of the model
            
        Returns:
        --------
        tuple
            (figure1, figure2) for ROC and PR curves
        """
        # Check if necessary metrics are available
        if 'roc_curve' not in metrics or 'pr_curve' not in metrics:
            print("ROC or PR curve data not available")
            return None
            
        # Plot ROC curve
        fig1, ax1 = plt.subplots(figsize=(10, 8))
        fpr, tpr, _ = metrics['roc_curve']
        ax1.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {metrics["roc_auc"]:.3f})')
        ax1.plot([0, 1], [0, 1], 'k--', lw=2)
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax1.legend(loc="lower right")
        ax1.grid(True)
        
        # Plot Precision-Recall curve
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        precision, recall, _ = metrics['pr_curve']
        ax2.plot(recall, precision, lw=2, label=model_name)
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curve')
        ax2.legend(loc="lower left")
        ax2.grid(True)
        
        # Plot F1 scores at different thresholds
        if 'f1_thresholds' in metrics:
            thresholds, f1_scores = zip(*metrics['f1_thresholds'])
            fig3, ax3 = plt.subplots(figsize=(10, 8))
            ax3.plot(thresholds, f1_scores, lw=2)
            if 'optimal_threshold' in metrics:
                ax3.axvline(x=metrics['optimal_threshold'], color='r', linestyle='--', 
                           label=f'Optimal threshold: {metrics["optimal_threshold"]:.3f}')
            ax3.set_xlim([0.0, 1.0])
            ax3.set_ylim([0.0, 1.05])
            ax3.set_xlabel('Threshold')
            ax3.set_ylabel('F1 Score')
            ax3.set_title('F1 Score vs. Threshold')
            ax3.legend(loc="best")
            ax3.grid(True)
            
            return fig1, fig2, fig3
        
        return fig1, fig2
    
    def ensemble_predictions(self, predictions_dict, method='majority_vote'):
        """
        Combine predictions from multiple models using ensemble methods
        
        Parameters:
        -----------
        predictions_dict : dict
            Dictionary of predictions from multiple models
        method : str
            Ensemble method to use ('majority_vote', 'weighted_vote', 'max_probability')
            
        Returns:
        --------
        numpy.ndarray
            Ensemble predictions
        """
        # Check if predictions_dict is None or empty
        if predictions_dict is None or not predictions_dict:
            print("Warning: No predictions to ensemble, returning empty array")
            return np.array([])
            
        # Filter out non-prediction keys
        model_names = [name for name in predictions_dict.keys() 
                    if not name.endswith('_prob') and not name.endswith('_score')]
        
        if not model_names:
            print("Warning: No valid model predictions found in predictions dictionary")
            # Return empty array or default predictions
            return np.array([])
            
        # Get predictions from each model
        predictions = []
        for name in model_names:
            if name in predictions_dict and predictions_dict[name] is not None:
                predictions.append(predictions_dict[name])
        
        if not predictions:
            print("Warning: No valid predictions to ensemble")
            return np.array([])
            
        # Stack predictions
        try:
            stacked_preds = np.column_stack(predictions)
        except ValueError as e:
            print(f"Error stacking predictions: {e}")
            # Try to handle predictions of different lengths
            min_len = min(len(p) for p in predictions)
            predictions = [p[:min_len] for p in predictions]
            stacked_preds = np.column_stack(predictions)
        
        # Apply ensemble method
        if method == 'majority_vote':
            # For each sample, count how many models predicted anomaly (1)
            anomaly_counts = np.sum(stacked_preds, axis=1)
            # If more than half of the models predict anomaly, consider it an anomaly
            ensemble_pred = np.where(anomaly_counts > len(model_names) / 2, 1, 0)
        elif method == 'weighted_vote':
            # Assign weights to models (can be customized based on model performance)
            weights = {}
            for name in model_names:
                if 'lstm' in name:
                    weights[name] = 2.0  # Give more weight to sequence models
                elif name in ['random_forest', 'xgboost']:
                    weights[name] = 1.5  # Give decent weight to supervised models
                else:
                    weights[name] = 1.0  # Default weight
            
            # Apply weights to predictions
            weighted_sum = np.zeros(len(stacked_preds))
            total_weight = 0
            for i, name in enumerate(model_names):
                if i < stacked_preds.shape[1]:  # Ensure index is in range
                    w = weights.get(name, 1.0)
                    weighted_sum += w * stacked_preds[:, i]
                    total_weight += w
            
            # Calculate threshold based on sum of weights
            threshold = total_weight / 2
            
            # Make final prediction
            ensemble_pred = np.where(weighted_sum > threshold, 1, 0)
        elif method == 'max_probability':
            # This method requires probability scores
            prob_names = [f"{name}_prob" for name in model_names if f"{name}_prob" in predictions_dict]
            score_names = [f"{name}_score" for name in model_names if f"{name}_score" in predictions_dict]
            
            if not prob_names and not score_names:
                print("Warning: No probability or score values available for ensemble")
                return np.where(np.sum(stacked_preds, axis=1) > 0, 1, 0)  # Fallback
                
            # Combine all available probabilities/scores
            combined_scores = []
            
            # Add probabilities (already between 0-1)
            for prob_name in prob_names:
                if prob_name in predictions_dict and predictions_dict[prob_name] is not None:
                    combined_scores.append(predictions_dict[prob_name])
                    
            # Add scores (need to normalize)
            for score_name in score_names:
                if score_name in predictions_dict and predictions_dict[score_name] is not None:
                    # Min-max scaling
                    scores = predictions_dict[score_name]
                    min_score, max_score = np.min(scores), np.max(scores)
                    if max_score > min_score:  # Avoid division by zero
                        normalized_scores = (scores - min_score) / (max_score - min_score)
                        combined_scores.append(normalized_scores)
                    else:
                        # If all scores are the same, use 0.5
                        combined_scores.append(np.ones(len(scores)) * 0.5)
                        
            # Stack all scores
            if combined_scores:
                try:
                    stacked_scores = np.column_stack(combined_scores)
                    # Take maximum score for each sample
                    max_scores = np.max(stacked_scores, axis=1)
                    # Make prediction based on threshold
                    ensemble_pred = np.where(max_scores > 0.5, 1, 0)
                except ValueError:
                    # Fallback if shapes don't match
                    print("Warning: Could not stack scores of different shapes")
                    return np.where(np.sum(stacked_preds, axis=1) > 0, 1, 0)
            else:
                # Fallback to majority vote if no scores available
                print("Warning: No valid scores available, using majority vote instead")
                return np.where(np.sum(stacked_preds, axis=1) > 0, 1, 0)
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
            
        return ensemble_pred

    def confidence_weighted_ensemble(self, predictions_dict):
        """
        Create an ensemble prediction that weights models by their confidence
        
        Parameters:
        -----------
        predictions_dict : dict
            Prediction results from predict_with_confidence
            
        Returns:
        --------
        numpy.ndarray, dict
            Ensemble predictions and confidence information
        """
        if 'confidence_scores' not in predictions_dict:
            return self.ensemble_predictions(predictions_dict), {}
        
        # Get the models and their confidence scores
        confidence_scores = predictions_dict['confidence_scores']
        model_names = list(confidence_scores.keys())
        
        if not model_names:
            return np.array([]), {}
        
        # Calculate weighted predictions
        weighted_sum = np.zeros(len(next(iter(confidence_scores.values()))))
        weight_sum = 0
        
        for model_name in model_names:
            if model_name in predictions_dict:
                predictions = predictions_dict[model_name]
                confidence = confidence_scores[model_name]
                
                # Weight predictions by confidence
                weighted_sum += predictions * confidence
                weight_sum += confidence
        
        # Normalize
        if weight_sum.any():
            # Avoid division by zero
            mask = weight_sum > 0
            weighted_avg = np.zeros_like(weighted_sum)
            weighted_avg[mask] = weighted_sum[mask] / weight_sum[mask]
            
            # Convert to binary predictions using dynamic threshold
            threshold = 0.6
            ensemble_pred = np.where(weighted_avg > threshold, 1, 0)
            
            return ensemble_pred, {
                'weighted_scores': weighted_avg,
                'threshold': threshold,
                'confidence_by_model': confidence_scores
            }
        else:
            return np.zeros(len(next(iter(confidence_scores.values())))), {}
    
    def calculate_dynamic_threshold(self, X_recent, model_name, window_size=100, percentile=95):
        """
        Calculate a dynamic threshold based on recent data
        
        Parameters:
        -----------
        X_recent : pandas.DataFrame
            Recent data for recalibrating thresholds
        model_name : str
            Name of the model to recalibrate
        window_size : int
            Number of recent samples to use
        percentile : float
            Percentile to use for threshold (higher = fewer anomalies)
        """
        # Use only the most recent data
        if len(X_recent) > window_size:
            X_recent = X_recent.iloc[-window_size:]
        
        # Prepare data
        if isinstance(X_recent, pd.DataFrame):
            X_recent = X_recent.fillna(0)
        
        # Get the appropriate scaler
        if model_name in self.scalers:
            scaler = self.scalers[model_name]
            X_scaled = scaler.transform(X_recent)
        else:
            X_scaled = X_recent
        
        # Calculate reconstruction errors
        if 'autoencoder' in model_name:
            model = self.models[model_name]
            reconstructions = model.predict(X_scaled)
            errors = np.mean(np.square(X_scaled - reconstructions), axis=1)
            
            # Set new threshold
            new_threshold = np.percentile(errors, percentile)
            self.thresholds[model_name] = new_threshold
            
            print(f"Updated {model_name} threshold: {new_threshold:.6f}")
        
        return self.thresholds.get(model_name)
        
    def save_models(self, base_filename='anomaly_detection_model'):
        """
        Save all trained models to disk
        
        Parameters:
        -----------
        base_filename : str
            Base filename to use for saved models
        """
        for name, model in self.models.items():
            # Save based on model type
            if 'autoencoder' in name or 'lstm' in name:
                # For Keras models
                model.save(f"{self.model_dir}/{base_filename}_{name}.h5")
            else:
                # For scikit-learn models
                joblib.dump(model, f"{self.model_dir}/{base_filename}_{name}.joblib")
                
        # Save scalers
        for name, scaler in self.scalers.items():
            joblib.dump(scaler, f"{self.model_dir}/{base_filename}_{name}_scaler.joblib")
            
        # Save thresholds if available
        if hasattr(self, 'thresholds'):
            np.save(f"{self.model_dir}/{base_filename}_thresholds.npy", self.thresholds)
            
        print(f"All models saved to {self.model_dir}")
    
    def load_models(self, base_filename='anomaly_detection_model'):
        """
        Load saved models from disk
        
        Parameters:
        -----------
        base_filename : str
            Base filename used when saving models
        """
        import os
        import glob
        
        # Load scikit-learn models
        joblib_files = glob.glob(f"{self.model_dir}/{base_filename}_*.joblib")
        for f in joblib_files:
            model_name = os.path.basename(f).replace(f"{base_filename}_", "").replace(".joblib", "")
            
            if model_name.endswith('_scaler'):
                # This is a scaler
                scaler_name = model_name.replace("_scaler", "")
                self.scalers[scaler_name] = joblib.load(f)
            else:
                # This is a model
                self.models[model_name] = joblib.load(f)
                
        # Load Keras models
        h5_files = glob.glob(f"{self.model_dir}/{base_filename}_*.h5")
        for f in h5_files:
            model_name = os.path.basename(f).replace(f"{base_filename}_", "").replace(".h5", "")
            self.models[model_name] = load_model(f)
            
        # Load thresholds if available
        threshold_file = f"{self.model_dir}/{base_filename}_thresholds.npy"
        if os.path.exists(threshold_file):
            self.thresholds = np.load(threshold_file, allow_pickle=True).item()
            
        print(f"Loaded {len(self.models)} models and {len(self.scalers)} scalers from {self.model_dir}")
        
    def get_model_summaries(self):
        """
        Get a summary of all trained models
        
        Returns:
        --------
        dict
            Dictionary with model information
        """
        summaries = {}
        
        for name, model in self.models.items():
            model_info = {
                'type': name,
                'has_scaler': name in self.scalers
            }
            
            if 'isolation_forest' in name:
                model_info['algorithm'] = 'Isolation Forest'
                model_info['n_estimators'] = model.n_estimators
                model_info['contamination'] = model.contamination
            elif 'one_class_svm' in name:
                model_info['algorithm'] = 'One-Class SVM'
                model_info['kernel'] = model.kernel
                model_info['nu'] = model.nu
            elif 'autoencoder' in name:
                model_info['algorithm'] = 'Autoencoder'
                model_info['threshold'] = self.thresholds.get(name, 'N/A')
            elif 'lstm' in name:
                model_info['algorithm'] = 'LSTM' + (' Autoencoder' if 'autoencoder' in name else ' Supervised')
                if 'autoencoder' in name:
                    model_info['threshold'] = self.thresholds.get(name, 'N/A')
            elif 'random_forest' in name:
                model_info['algorithm'] = 'Random Forest'
                if hasattr(model, 'named_steps') and 'classifier' in model.named_steps:
                    model_info['n_estimators'] = model.named_steps['classifier'].n_estimators
            elif 'xgboost' in name:
                model_info['algorithm'] = 'XGBoost'
                if hasattr(model, 'named_steps') and 'classifier' in model.named_steps:
                    model_info['n_estimators'] = model.named_steps['classifier'].n_estimators
                    
            summaries[name] = model_info
            
        return summaries

# Example usage of the Advanced Anomaly Detection class
def run_anomaly_detection_demo():
    """Run a demonstration of the advanced anomaly detection system"""
    # Create a detector instance
    detector = AdvancedAnomalyDetection(model_dir='models')
    
    # Generate or load data
    # This would come from your TimeSeriesFeatureEngineering pipeline
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=1000, 
        n_features=20, 
        n_informative=10, 
        n_redundant=5, 
        n_clusters_per_class=2, 
        weights=[0.95, 0.05],  # Imbalanced dataset with 5% anomalies
        random_state=42
    )
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    y = pd.Series(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train multiple models
    detector.train_isolation_forest(X_train, contamination=0.05)
    detector.train_one_class_svm(X_train, nu=0.05)
    detector.train_autoencoder(X_train, epochs=20, batch_size=32)
    detector.train_supervised_rf(X_train, y_train)
    detector.train_supervised_xgboost(X_train, y_train)
    
    # Evaluate models
    for model_name in detector.models.keys():
        metrics = detector.evaluate_model(X_test, y_test, model_name)
        detector.plot_evaluation_curves(metrics, model_name)
    
    # Make ensemble prediction
    predictions = detector.predict_anomalies(X_test)
    ensemble_pred = detector.ensemble_predictions(predictions, method='weighted_vote')
    
    # Evaluate ensemble
    from sklearn.metrics import classification_report
    print("\nEnsemble Model Results:")
    print(classification_report(y_test, ensemble_pred))
    
    # Save models
    detector.save_models()
    
    return detector

if __name__ == "__main__":
    detector = run_anomaly_detection_demo()