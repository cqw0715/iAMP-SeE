import numpy as np
import pandas as pd
import os
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, roc_curve
from sklearn.utils.class_weight import compute_class_weight
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

# ---------------------- 1. Data Loading and Preprocessing ----------------------
def load_data(csv_path):
    data = pd.read_csv(csv_path)
    print(f"Class distribution:\n{data['label'].value_counts()}")
    return data['sequence'].values, data['label'].values

# ---------------------- 2. ESM Feature Extraction ----------------------
def load_esm_model():
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    return model.eval(), batch_converter

esm_model, esm_batch_converter = load_esm_model()

def get_esm_features(sequences, cache_path='esm_features.pkl', batch_size=8):
    """Extract features using ESM-2 model with caching support"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if os.path.exists(cache_path):
        print(f"Loading ESM features from cache file {cache_path}...")
        with open(cache_path, 'rb') as f:
            features = pickle.load(f)
        print(f"Loaded {len(features)} features from cache.")
        return features
    
    print("Cache file not found, starting ESM feature extraction...")
    print(f"Total sequences to process: {len(sequences)}")
    features = []
    
    for i in range(0, len(sequences), batch_size):
        batch_start = i
        batch_end = min(i + batch_size, len(sequences))
        batch = sequences[batch_start:batch_end]
        
        print(f"\nProcessing batch {i//batch_size + 1}/{(len(sequences)-1)//batch_size + 1}")
        print(f"Sequence indices: {batch_start}-{batch_end-1}")
        
        batch_data = [(str(idx), seq) for idx, seq in enumerate(batch)]
        print("Converting sequences to ESM format...")
        batch_labels, batch_strs, batch_tokens = esm_batch_converter(batch_data)
        print(f"Batch tokens shape: {batch_tokens.shape}")
        
        batch_tokens = batch_tokens.to(device)
        esm_model.to(device)
        
        print("Extracting features with ESM model...")
        with torch.no_grad():
            results = esm_model(batch_tokens, repr_layers=[33], return_contacts=False)
            token_representations = results["representations"][33]
            print(f"Token representations shape: {token_representations.shape}")
            
            seq_lengths = (batch_tokens != esm_model.alphabet.padding_idx).sum(1)
            print(f"Sequence lengths: {seq_lengths}")
            
            for seq_idx in range(token_representations.size(0)):
                seq_len = seq_lengths[seq_idx]
                seq_rep = token_representations[seq_idx, :seq_len]
                mean_rep = seq_rep.mean(0).cpu().numpy()
                features.append(mean_rep)
                print(f"Sequence {batch_start + seq_idx}: Processed {seq_len} tokens -> Feature vector shape: {mean_rep.shape}")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("Cleared GPU memory")
    
    features = np.array(features)
    print(f"\nFinal features array shape: {features.shape}")
    
    print(f"Saving features to cache file {cache_path}...")
    with open(cache_path, 'wb') as f:
        pickle.dump(features, f)
    print(f"ESM features saved successfully to {cache_path}")
    
    return features


class ProteinDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.features[idx]), torch.LongTensor([self.labels[idx]])

# ---------------------- 3. SeE Attention Mechanism Module ----------------------
class SEAttention(Layer):
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

# ---------------------- 4. Model Architecture ----------------------
def build_double_attention_model(input_dim):
    esm_input = Input(shape=(input_dim,))
    
    x = Reshape((1, input_dim))(esm_input)
    
    x = Conv1D(256, 5, padding='same', activation='swish')(x)
    x = BatchNormalization()(x)
    x = Conv1D(128, 3, padding='same', activation='swish')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = Bidirectional(LSTM(256, return_sequences=True))(x)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    
    x = SEAttention(channels=256)(x)  
    x = ECAAttention(kernel_size=3)(x) 
    
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
    output = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=esm_input, outputs=output)
    model.compile(optimizer=Adam(learning_rate=1e-4),
                loss='binary_crossentropy',
                metrics=['accuracy'])
    return model

# ---------------------- 5. Training Pipeline ----------------------
def enhanced_cross_validation(features, labels, n_splits=10):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = []
    roc_data = []
    
    os.makedirs('roc_data', exist_ok=True)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(features)):
        print(f'\n--- Fold {fold+1} ---')
        X_train, X_val = features[train_idx], features[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]
        
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weights = {i:w for i,w in enumerate(class_weights)}
        
        model = build_double_attention_model(features.shape[1])
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
        ]
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=30,
            batch_size=32,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        y_pred = model.predict(X_val).flatten()
        y_pred_class = (y_pred > 0.5).astype(int)
        
        metrics = {
            'Fold': fold+1,
            'Accuracy': accuracy_score(y_val, y_pred_class),
            'Sensitivity': recall_score(y_val, y_pred_class),
            'Specificity': recall_score(y_val, y_pred_class, pos_label=0),
            'MCC': matthews_corrcoef(y_val, y_pred_class),
            'F1': f1_score(y_val, y_pred_class),
            'AUC': roc_auc_score(y_val, y_pred)
        }
        results.append(metrics)
        
        fpr, tpr, _ = roc_curve(y_val, y_pred)
        roc_data.append((fpr, tpr, metrics['AUC']))
        
        roc_df = pd.DataFrame({
            'FPR': fpr,
            'TPR': tpr
        })
        roc_df.to_csv(f'roc_data/fold_{fold+1}_roc_data.csv', index=False)
        print(f"Saved ROC data for fold {fold+1} to roc_data/fold_{fold+1}_roc_data.csv")
        
        del model
        tf.keras.backend.clear_session()
    
    save_average_roc_data(roc_data)
    
    return pd.DataFrame(results), roc_data

def save_average_roc_data(roc_data):
    max_length = max(len(fpr) for fpr, tpr, _ in roc_data)
    
    interp_fpr = np.linspace(0, 1, max_length)
    interp_tprs = []
    
    for fpr, tpr, auc in roc_data:
        interp_tpr = np.interp(interp_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        interp_tpr[-1] = 1.0
        interp_tprs.append(interp_tpr)
    
    mean_tpr = np.mean(interp_tprs, axis=0)
    std_tpr = np.std(interp_tprs, axis=0)
    
    avg_roc_df = pd.DataFrame({
        'FPR': interp_fpr,
        'Mean_TPR': mean_tpr,
        'TPR_Std': std_tpr,
        'TPR_Upper': np.minimum(mean_tpr + std_tpr, 1),
        'TPR_Lower': np.maximum(mean_tpr - std_tpr, 0)
    })
    
    avg_roc_df.to_csv('roc_data/average_roc_data.csv', index=False)
    print("Saved average ROC data to roc_data/average_roc_data.csv")
    
    auc_values = [auc for _, _, auc in roc_data]
    auc_df = pd.DataFrame({
        'Fold': range(1, len(auc_values) + 1),
        'AUC': auc_values
    })
    auc_df.to_csv('roc_data/auc_values.csv', index=False)
    print("Saved AUC values to roc_data/auc_values.csv")

# ---------------------- 6. Result Visualization ----------------------
def plot_roc_curves(roc_data):
    plt.figure(figsize=(10, 8))
    
    for i, (fpr, tpr, auc) in enumerate(roc_data):
        plt.plot(fpr, tpr, alpha=0.3, lw=1, label=f'Fold {i+1} (AUC = {auc:.3f})')
    
    avg_roc_df = pd.read_csv('roc_data/average_roc_data.csv')
    mean_auc = np.mean([auc for _, _, auc in roc_data])
    std_auc = np.std([auc for _, _, auc in roc_data])
    
    plt.plot(avg_roc_df['FPR'], avg_roc_df['Mean_TPR'], color='b', 
             label=f'Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})', lw=2)
    
    plt.fill_between(avg_roc_df['FPR'], 
                     avg_roc_df['TPR_Lower'], 
                     avg_roc_df['TPR_Upper'], 
                     color='grey', alpha=0.2)
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves with Mean and Standard Deviation')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig('iAMP-SeE.png', dpi=300, bbox_inches='tight')
    plt.show()

# ---------------------- Main Pipeline ----------------------
if __name__ == "__main__":
    # Load data
    sequences, labels = load_data("data1.csv")
    
    # Extract features using ESM
    print("\nExtracting ESM features...")
    esm_features = get_esm_features(sequences)
    print(f"ESM features shape: {esm_features.shape}")
    
    # Perform cross-validation
    results_df, roc_data = enhanced_cross_validation(esm_features, labels)
    
    # Result analysis
    final_metrics = results_df.mean(numeric_only=True).to_dict()
    final_stds = results_df.std(numeric_only=True).to_dict()
    
    print("\n=== Final Average Metrics ===")
    for k in final_metrics.keys():
        print(f"{k}: {final_metrics[k]:.4f} ± {final_stds[k]:.4f}")
    
    # Save results
    results_df.to_csv("iAMP-SeE.csv", index=False)
    
    # Plot ROC curves
    plot_roc_curves(roc_data)
