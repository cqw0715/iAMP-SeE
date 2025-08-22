import numpy as np
import pandas as pd
import os
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, label_ranking_average_precision_score, roc_auc_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Embedding, Conv1D, Bidirectional, LSTM, Concatenate, Layer, GlobalAveragePooling1D, Reshape, Lambda, BatchNormalization, GlobalMaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
from collections import defaultdict
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import esm
import pickle
from sklearn.neighbors import NearestNeighbors

def get_sequence_features(sequences):
    """Extract basic sequence features including amino acid composition, length, and dipeptide frequencies"""
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    
    features = []
    for seq in sequences:
        # Calculate amino acid counts and composition
        counts = {aa: seq.count(aa) for aa in amino_acids}
        total = max(1, len(seq))
        composition = [counts[aa]/total for aa in amino_acids]
        
        # Normalized sequence length feature
        length_feature = [len(seq)/1000]
        
        # Calculate dipeptide frequencies
        dipeptides = [seq[i:i+2] for i in range(len(seq)-1)]
        dipeptide_counts = {dp: dipeptides.count(dp) for dp in set(dipeptides)}
        total_dp = max(1, len(dipeptides))
        dipeptide_feature = [dipeptide_counts.get(dp, 0)/total_dp for dp in ['AA','AC','AD','AE','AG']]
        
        # Combine all features
        combined = composition + length_feature + dipeptide_feature
        features.append(combined)
    
    return np.array(features)

def robust_oversampling(sequences, labels):
    """Perform oversampling with fallback strategies if primary methods fail"""
    try:
        # First try ADASYN oversampling
        features = get_sequence_features(sequences)
        ada = ADASYN(random_state=42, n_neighbors=min(5, len(sequences)-1))
        features_resampled, labels_resampled = ada.fit_resample(features, labels)
        
        # Map resampled features back to original sequences
        nbrs = NearestNeighbors(n_neighbors=1).fit(features)
        _, indices = nbrs.kneighbors(features_resampled)
        sequences_resampled = sequences[indices.flatten()]
        
        print("Successfully used ADASYN for oversampling")
        return sequences_resampled, labels_resampled
    except ValueError as e:
        print(f"ADASYN failed: {str(e)}. Trying SMOTE...")
    
    try:
        # Fallback to SMOTE if ADASYN fails
        features = get_sequence_features(sequences)
        smote = SMOTE(random_state=42, k_neighbors=min(5, len(sequences)-1))
        features_resampled, labels_resampled = smote.fit_resample(features, labels)
        
        # Map resampled features back to original sequences
        nbrs = NearestNeighbors(n_neighbors=1).fit(features)
        _, indices = nbrs.kneighbors(features_resampled)
        sequences_resampled = sequences[indices.flatten()]
        
        print("Successfully used SMOTE for oversampling")
        return sequences_resampled, labels_resampled
    except ValueError as e:
        print(f"SMOTE failed: {str(e)}. Using RandomOverSampler as final fallback...")
    
    # Final fallback to simple random oversampling
    ros = RandomOverSampler(random_state=42)
    seq_reshaped = np.array(sequences).reshape(-1, 1)
    seq_resampled, labels_resampled = ros.fit_resample(seq_reshaped, labels)
    sequences_resampled = seq_resampled.ravel()
    
    print("Used RandomOverSampler as fallback")
    return sequences_resampled, labels_resampled

def load_and_oversample_data(csv_path):
    """Load data from CSV and perform oversampling to balance classes"""
    data = pd.read_csv(csv_path)
    print(f"Original class distribution:\n{data['label'].value_counts()}")
    
    sequences = data['sequence'].values
    labels = data['label'].values
    
    # Apply robust oversampling
    sequences_resampled, labels_resampled = robust_oversampling(sequences, labels)
    
    print(f"\nClass distribution after oversampling:\n{pd.Series(labels_resampled).value_counts()}")
    return sequences_resampled, labels_resampled

def load_esm_model():
    """Load pretrained ESM-2 model and batch converter"""
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    return model.eval(), batch_converter

# Load ESM model globally for feature extraction
esm_model, esm_batch_converter = load_esm_model()

def get_esm_features(sequences, cache_path='esm_features_2_1.0.pkl', batch_size=8):
    """Extract ESM embeddings for sequences with caching"""
    if os.path.exists(cache_path):
        print(f"Loading ESM features from cache file {cache_path}...")
        with open(cache_path, 'rb') as f:
            features = pickle.load(f)
        return features
    
    print("Cache file not found, starting ESM feature extraction...")
    features = []
    
    # Process sequences in batches
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i+batch_size]
        
        # Convert batch to ESM format
        batch_data = [(str(i), seq) for i, seq in enumerate(batch)]
        batch_labels, batch_strs, batch_tokens = esm_batch_converter(batch_data)
        
        # Extract embeddings
        with torch.no_grad():
            results = esm_model(batch_tokens, repr_layers=[33], return_contacts=False)
            token_representations = results["representations"][33]
            
            # Calculate mean pooling of embeddings
            seq_lengths = (batch_tokens != esm_model.alphabet.padding_idx).sum(1)
            for seq_idx in range(token_representations.size(0)):
                seq_len = seq_lengths[seq_idx]
                seq_rep = token_representations[seq_idx, :seq_len]
                features.append(seq_rep.mean(0).cpu().numpy())
        
        # Clear memory if using GPU
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    features = np.array(features)
    
    # Cache the extracted features
    with open(cache_path, 'wb') as f:
        pickle.dump(features, f)
    print(f"ESM features saved to {cache_path}")
    
    return features

class ProteinDataset(Dataset):
    """PyTorch Dataset class for protein sequence features and labels"""
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.features[idx]), torch.LongTensor([self.labels[idx]])

class SEAttention(Layer):
    """Squeeze-and-Excitation attention layer"""
    def __init__(self, channels, reduction=16):
        super(SEAttention, self).__init__()
        self.global_avg = GlobalAveragePooling1D()
        self.fc1 = Dense(channels // reduction, activation='relu')
        self.fc2 = Dense(channels, activation='sigmoid')
        self.reshape = Reshape((1, channels))
        
    def call(self, inputs):
        se = self.global_avg(inputs)
        se = self.fc1(se)
        se = self.fc2(se)
        se = self.reshape(se)
        return inputs * se

class ECAAttention(Layer):
    """Efficient Channel Attention layer"""
    def __init__(self, kernel_size=3, **kwargs):
        super(ECAAttention, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.gap = GlobalAveragePooling1D(keepdims=True)
        
    def build(self, input_shape):
        channels = input_shape[-1]
        self.conv1d = Conv1D(1, 
                            kernel_size=self.kernel_size, 
                            padding='same',
                            use_bias=False,
                            kernel_initializer='he_normal')
        super().build(input_shape)
        
    def call(self, inputs):
        y = self.gap(inputs)
        y = self.conv1d(y)
        y = tf.sigmoid(y)
        return inputs * y

def build_double_attention_model(input_dim, num_classes=5):
    """Build neural network model with both SE and ECA attention mechanisms"""
    # Input layer for ESM features
    esm_input = Input(shape=(input_dim,))
    
    # Reshape for 1D operations
    x = Reshape((1, input_dim))(esm_input)
    
    # Initial convolutional blocks
    x = Conv1D(256, 5, padding='same', activation='swish')(x)
    x = BatchNormalization()(x)
    x = Conv1D(128, 3, padding='same', activation='swish')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    # Bidirectional LSTM layers
    x = Bidirectional(LSTM(256, return_sequences=True))(x)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    
    # Dual attention mechanisms
    x = SEAttention(channels=256)(x)
    x = ECAAttention(kernel_size=3)(x)
    
    # Pooling and dense layers
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    x = Concatenate()([avg_pool, max_pool])
    
    x = Dense(512, activation='swish', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='swish', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='swish', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='swish', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation='softmax')(x)
    
    # Compile model
    model = Model(inputs=esm_input, outputs=output)
    model.compile(optimizer=Adam(learning_rate=1e-4),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    return model

def enhanced_cross_validation(features, labels, n_splits=10):
    """Perform k-fold cross validation with comprehensive metrics tracking"""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = []
    class_metrics = defaultdict(list)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(features)):
        print(f'\n--- Fold {fold+1} ---')
        X_train, X_val = features[train_idx], features[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]
        
        # Build and train model
        model = build_double_attention_model(features.shape[1], num_classes=5)
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
        ]
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=120,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Make predictions
        y_pred = model.predict(X_val)
        y_pred_class = np.argmax(y_pred, axis=1)
        y_score = y_pred
        
        # Calculate metrics
        metrics = {
            'Fold': fold+1,
            'ACC': accuracy_score(y_val, y_pred_class),
            'F1_Micro': f1_score(y_val, y_pred_class, average='micro'),
            'F1_Macro': f1_score(y_val, y_pred_class, average='macro'),
            'HL': hamming_loss(y_val, y_pred_class),
            'MAP': label_ranking_average_precision_score(
                tf.one_hot(y_val, depth=5).numpy(), 
                y_score
            ),
            'ROC_AUC_OVR': roc_auc_score(
                tf.one_hot(y_val, depth=5).numpy(),
                y_score,
                multi_class='ovr'
            ),
            'ROC_AUC_OVO': roc_auc_score(
                tf.one_hot(y_val, depth=5).numpy(),
                y_score,
                multi_class='ovo'
            )
        }
        results.append(metrics)
        
        # Calculate per-class metrics
        cm = confusion_matrix(y_val, y_pred_class)
        class_acc = cm.diagonal() / cm.sum(axis=1)
        
        y_val_onehot = tf.one_hot(y_val, depth=5).numpy()
        class_roc_auc = []
        for i in range(5):
            if len(np.unique(y_val_onehot[:, i])) > 1:
                class_roc_auc.append(roc_auc_score(y_val_onehot[:, i], y_score[:, i]))
            else:
                class_roc_auc.append(np.nan)
        
        for i in range(5):
            class_metrics[f'Class_{i}_ACC'].append(class_acc[i])
            class_metrics[f'Class_{i}_ROC_AUC'].append(class_roc_auc[i])
        
        # Clean up memory
        del model
        tf.keras.backend.clear_session()
    
    # Combine all results
    results_df = pd.DataFrame(results)
    for k, v in class_metrics.items():
        results_df[k] = v
    
    return results_df

if __name__ == "__main__":
    # Main execution block
    sequences, labels = load_and_oversample_data("All_2_16200.csv")
    
    print("\nExtracting ESM features...")
    esm_features = get_esm_features(sequences)
    print(f"ESM features shape: {esm_features.shape}")
    
    # Perform cross-validation
    results_df = enhanced_cross_validation(esm_features, labels)
    
    # Calculate final metrics
    final_metrics = results_df.mean(numeric_only=True).to_dict()
    final_stds = results_df.std(numeric_only=True).to_dict()
    
    # Print results
    print("\n=== Final Average Metrics ===")
    for k in final_metrics.keys():
        print(f"{k}: {final_metrics[k]:.4f} ± {final_stds[k]:.4f}")
    
    print("\n=== Performance Metrics per Class ===")
    for i in range(5):
        acc_mean = final_metrics[f'Class_{i}_ACC']
        acc_std = final_stds[f'Class_{i}_ACC']
        roc_mean = final_metrics[f'Class_{i}_ROC_AUC']
        roc_std = final_stds[f'Class_{i}_ROC_AUC']
        print(f"Class {i}: ACC = {acc_mean:.4f} ± {acc_std:.4f}, ROC AUC = {roc_mean:.4f} ± {roc_std:.4f}")
    
    # Save results
    results_df.to_csv("iAMP-SeE.csv", index=False)
    
    with open("iAMP-SeE_results.txt", "w") as f:
        f.write("=== Final Average Metrics ===\n")
        for k in final_metrics.keys():
            if not k.startswith('Class_'):
                f.write(f"{k}: {final_metrics[k]:.4f} ± {final_stds[k]:.4f}\n")
        
        f.write("\n=== Performance Metrics per Class ===\n")
        for i in range(5):
            acc_mean = final_metrics[f'Class_{i}_ACC']
            acc_std = final_stds[f'Class_{i}_ACC']
            roc_mean = final_metrics[f'Class_{i}_ROC_AUC']
            roc_std = final_stds[f'Class_{i}_ROC_AUC']
            f.write(f"Class {i}: ACC = {acc_mean:.4f} ± {acc_std:.4f}, ROC AUC = {roc_mean:.4f} ± {roc_std:.4f}\n")
        
        f.write("\n=== Detailed Cross-Validation Results ===\n")
        f.write(results_df.to_string())
    
    print("\nAll results have been saved to iAMP-SeE.csv and iAMP-SeE_results.txt")