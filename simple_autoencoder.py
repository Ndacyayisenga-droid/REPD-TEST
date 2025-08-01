#!/usr/bin/env python3
"""
Simple Autoencoder implementation compatible with TensorFlow 2.x
for use with REPD model training.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class SimpleAutoencoder:
    """A simple autoencoder using PCA for dimensionality reduction"""

    def __init__(self, n_components=None):
        self.n_components = n_components
        self.pca = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def fit(self, X):
        """Fit the autoencoder on the data"""
        # Scale the data
        X_scaled = self.scaler.fit_transform(X)
        
        # If n_components not specified, use 80% of variance
        if self.n_components is None:
            self.n_components = max(1, int(X.shape[1] * 0.8))
        
        # Ensure n_components doesn't exceed available dimensions
        max_components = min(X.shape[0] - 1, X.shape[1])
        self.n_components = min(self.n_components, max_components)
        
        # Create and fit PCA
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(X_scaled)
        self.is_fitted = True
        return self
    
    def transform(self, X):
        """Transform data to latent space"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before transform")
        X_scaled = self.scaler.transform(X)
        return self.pca.transform(X_scaled)
    
    def inverse_transform(self, X):
        """Transform data back from latent space"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before inverse_transform")
        reconstructed = self.pca.inverse_transform(X)
        return self.scaler.inverse_transform(reconstructed)
    
    def fit_transform(self, X):
        """Fit and transform in one step"""
        return self.fit(X).transform(X)

class DeepAutoencoder:
    """Deep autoencoder using multiple PCA layers"""
    
    def __init__(self, compression_ratio=0.5):
        self.compression_ratio = compression_ratio
        self.encoders = []
        self.decoders = []
        self.scalers = []
        self.is_fitted = False
        self.layers = None  # Will be set during fit
        
    def fit(self, X):
        """Fit the deep autoencoder"""
        current_X = X.copy()
        input_dim = X.shape[1]
        
        # Automatically determine layer sizes based on input dimension
        bottleneck_dim = max(1, int(input_dim * self.compression_ratio))
        self.layers = [
            input_dim,
            max(1, input_dim // 2),
            bottleneck_dim,
            max(1, input_dim // 2),
            input_dim
        ]
        
        # Create encoders (dimensionality reduction)
        for i in range(len(self.layers) - 1):
            # Ensure n_components doesn't exceed available dimensions
            n_components = min(self.layers[i], self.layers[i + 1], 
                             min(X.shape[0] - 1, X.shape[1]))
            if n_components < 1:
                n_components = 1
                
            encoder = PCA(n_components=n_components)
            scaler = StandardScaler()
            
            # Scale and fit
            current_X_scaled = scaler.fit_transform(current_X)
            encoded = encoder.fit_transform(current_X_scaled)
            
            self.encoders.append(encoder)
            self.scalers.append(scaler)
            current_X = encoded
        
        # Create decoders (dimensionality expansion)
        for i in range(len(self.layers) - 2, -1, -1):
            n_components = min(self.layers[i], min(X.shape[0] - 1, X.shape[1]))
            if n_components < 1:
                n_components = 1
            decoder = PCA(n_components=n_components)
            decoder.fit(self.scalers[i].transform(X))
            self.decoders.append(decoder)
        
        self.is_fitted = True
        return self
    
    def transform(self, X):
        """Transform through encoder"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before transform")
        
        current_X = X.copy()
        for encoder, scaler in zip(self.encoders, self.scalers):
            current_X_scaled = scaler.transform(current_X)
            current_X = encoder.transform(current_X_scaled)
        
        return current_X
    
    def inverse_transform(self, X):
        """Transform through decoder"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before inverse_transform")
        
        current_X = X.copy()
        for decoder in reversed(self.decoders):
            current_X = decoder.inverse_transform(current_X)
        
        return current_X

class ConvolutionalAutoencoder:
    """Convolutional-like autoencoder using PCA with reshaping"""
    
    def __init__(self, input_dim=None):
        self.input_dim = input_dim
        self.pca = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def fit(self, X):
        """Fit the convolutional autoencoder"""
        # Reshape data to simulate convolutional structure
        if self.input_dim is None:
            self.input_dim = X.shape[1]
        
        # Scale the data
        X_scaled = self.scaler.fit_transform(X)
        
        # Use PCA with fewer components for "convolutional" effect
        n_components = max(1, int(self.input_dim * 0.6))
        # Ensure n_components doesn't exceed available dimensions
        max_components = min(X.shape[0] - 1, X.shape[1])
        n_components = min(n_components, max_components)
        
        self.pca = PCA(n_components=n_components)
        self.pca.fit(X_scaled)
        self.is_fitted = True
        return self
    
    def transform(self, X):
        """Transform data to latent space"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before transform")
        X_scaled = self.scaler.transform(X)
        return self.pca.transform(X_scaled)
    
    def inverse_transform(self, X):
        """Transform data back from latent space"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before inverse_transform")
        reconstructed = self.pca.inverse_transform(X)
        return self.scaler.inverse_transform(reconstructed) 