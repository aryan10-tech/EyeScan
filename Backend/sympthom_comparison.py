import pandas as pd
import spacy

# Load the Excel file
df = pd.read_excel('Symptoms.xlsx')

# Extract symptom descriptions for each disease
disease_symptom_descriptions = []
for i in range(1, 11):  # Assuming there are 10 diseases
    symptom_columns = [f'Sympt{j}' for j in range(1, 11)]
    symptoms = df.loc[df['ID'] == i, symptom_columns].values.flatten()
    symptoms = [str(symptom).lower() for symptom in symptoms if not pd.isna(symptom)]
    symptom_descriptions_processed = ' '.join(symptoms)
    disease_symptom_descriptions.append(symptom_descriptions_processed)

# Load pre-trained word vectors model
nlp = spacy.load("en_core_web_md")

# Encode symptom descriptions into vectors
disease_vectors = [nlp(symptom) for symptom in disease_symptom_descriptions]

def find_disease(input_symptoms):
    # Encode input symptoms into vectors
    input_vectors = [nlp(symptom.lower()) for symptom in input_symptoms]
    
    # Calculate similarity between input symptoms and each symptom description for all diseases
    avg_similarities = []
    for disease_vector in disease_vectors:
        similarities = [input_vector.similarity(disease_vector) for input_vector in input_vectors]
        avg_similarity = sum(similarities) / len(similarities)
        avg_similarities.append(avg_similarity)
        #Low accuracy which is reducing the efficiency of the whole model.
        print(avg_similarity)
    # Find the disease with the highest average similarity
    # Find the index of the disease with the maximum average similarity
    max_similarity_index = max(range(len(avg_similarities)), key=lambda i: avg_similarities[i])
    print(max_similarity_index)
    
    # Return the disease corresponding to the most similar disease
    print(avg_similarities[max_similarity_index])
    disease = df.loc[df['ID'] == max_similarity_index + 1, 'Disease'].values[0]
    description = df.loc[df['ID'] == max_similarity_index + 1, 'Description'].values[0]

    

    return {
        'Disease': disease,
        'Description': description
    }
    
    
    

# Test the function with multiple input symptoms
#input_symptoms = ['sickle cell retinopathy','pain','fatigue', 'i have red eyes',]
#result = find_disease(input_symptoms)
#print(result)
