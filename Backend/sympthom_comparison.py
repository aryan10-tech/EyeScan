import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Dense, Embedding, LSTM, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Load the Excel file
df = pd.read_excel('Symptoms.xlsx')

# Extract symptom descriptions for each disease
disease_symptom_descriptions = []
disease_ids = df['ID'].unique()
for disease_id in disease_ids:
    symptom_columns = [f'Sympt{j}' for j in range(1, 11)]
    symptoms = df.loc[df['ID'] == disease_id, symptom_columns].values.flatten()
    symptoms = [str(symptom).lower() for symptom in symptoms if not pd.isna(symptom)]
    symptom_descriptions_processed = ' '.join(symptoms)
    disease_symptom_descriptions.append(symptom_descriptions_processed)

# Tokenize the descriptions
tokenizer = Tokenizer()
tokenizer.fit_on_texts(disease_symptom_descriptions)
sequences = tokenizer.texts_to_sequences(disease_symptom_descriptions)
word_index = tokenizer.word_index

# Pad sequences
max_len = max(len(seq) for seq in sequences)
data = pad_sequences(sequences, maxlen=max_len, padding='post')

# Create labels
labels = np.arange(1, len(disease_ids) + 1)

# Split data
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

def siamese_lstm_model(input_shape, embedding_dim, vocabulary_size):
    input = Input(shape=input_shape)
    embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim)(input)
    lstm_layer = Bidirectional(LSTM(64))(embedding)
    model = Model(inputs=input, outputs=lstm_layer)
    return model

embedding_dim = 100
vocabulary_size = len(word_index) + 1
input_shape = (max_len,)

# Base network
base_network = siamese_lstm_model(input_shape, embedding_dim, vocabulary_size)

# Inputs for the two branches
input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)

# Outputs from the two branches
processed_a = base_network(input_a)
processed_b = base_network(input_b)

# Lambda layer to compute the absolute difference
distance = Lambda(lambda x: tf.abs(x[0] - x[1]))([processed_a, processed_b])
output = Dense(1, activation='sigmoid')(distance)

model = Model(inputs=[input_a, input_b], outputs=output)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Generate pairs for training
def generate_pairs(data, labels):
    pairs = []
    targets = []
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            pairs += [[data[i], data[j]]]
            if labels[i] == labels[j]:
                targets += [1]
            else:
                targets += [0]
    return np.array(pairs), np.array(targets)

pairs, targets = generate_pairs(train_data, train_labels)
pairs_test, targets_test = generate_pairs(test_data, test_labels)

# Train the model
model.fit([pairs[:, 0], pairs[:, 1]], targets, batch_size=32, epochs=10, validation_data=([pairs_test[:, 0], pairs_test[:, 1]], targets_test))

def find_disease(input_symptoms):
    input_sequences = tokenizer.texts_to_sequences([input_symptoms])
    input_data = pad_sequences(input_sequences, maxlen=max_len, padding='post')

    similarities = []
    for disease_desc in data:
        disease_desc_exp = np.expand_dims(disease_desc, axis=0)

        # Predict similarity (removed extra dimension expansion)
        similarity = model.predict([input_data, disease_desc_exp])
        similarities.append(np.mean(similarity))

    max_similarity_index = np.argmax(similarities)
    disease = df.loc[df['ID'] == disease_ids[max_similarity_index], 'Disease'].values[0]
    description = df.loc[df['ID'] == disease_ids[max_similarity_index], 'Description'].values[0]

    return {
        'Disease': disease,
        'Description': description
    }

# Test the function with multiple input symptoms
#input_symptoms = 'sickle cell retinopathy, pain fatigue, red eyes'
#result = find_disease(input_symptoms)
#print(result)
