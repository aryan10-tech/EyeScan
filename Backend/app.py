import requests
from bs4 import BeautifulSoup
from flask import Flask, render_template, request,jsonify
from Ocular_main import processed_img
from sympthom_comparison import find_disease  # Import the function for symptom comparison

app = Flask(__name__,template_folder='../Frontend/templates')
#run the app by using the command (flask run)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/scrape')
def scrape_disease_information():
    # Define the URL to scrape
    url = 'https://en.wikipedia.org/wiki/'  # Wikipedia URL
    
    # Get the disease name from the query parameter
    disease_name = request.args.get('disease')

    # Perform web scraping
    response = requests.get(url + disease_name)
    print(url+disease_name)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        # Extract disease information from the scraped page
        paragraphs = soup.find_all('p')
        if len(paragraphs) > 1:  # Check if there are at least two paragraphs
            description = paragraphs[1].text  # Extract text from the second paragraph
            return jsonify({'description': description})
        else:
            return jsonify({'error': 'Description not found on the website'})
    else:
        return jsonify({'error': 'Failed to fetch data from the website'})

@app.route('/submit', methods=['POST'])
def submit():
    symptoms = request.form.getlist('symptoms')
    image1 = request.files['image1']
    image2 = request.files['image2']
    
    # Save the images to a temporary location
    image1_path = 'temp_image1.jpg'
    image2_path = 'temp_image2.jpg'
    image1.save(image1_path)
    image2.save(image2_path)
    
    # Process the images
    result = processed_img(image1_path)
    print(result)
    result1 = processed_img(image2_path)
    print(result1)
    symptoms.append(result)
    symptoms.append(result1)
    print(symptoms)

    # Remove None values from the symptoms list
    symptoms = [symptom for symptom in symptoms if symptom is not None]

    if not symptoms:
        return "No valid symptoms provided."
    
    # Call the function to find the most likely disease based on symptoms
    most_likely_disease = find_disease(symptoms)
    
    return render_template('result.html', result=result, most_likely_disease=most_likely_disease)

if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0')
