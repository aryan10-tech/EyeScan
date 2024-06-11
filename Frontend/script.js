// Function to fetch disease information dynamically
function fetchDiseaseInformation() {
    // Make a request to the backend server or API endpoint to fetch disease information
    fetch('/search?query=disease') // Replace '/search?query=disease' with the actual endpoint URL
        .then(response => response.json())
        .then(data => {
            // Update the HTML elements with the fetched disease information
            document.getElementById('disease').textContent = `Disease: ${data.name}`;
            document.getElementById('description').textContent = `Description: ${data.description}`;
        })
        .catch(error => {
            console.error('Error fetching disease information:', error);
            document.getElementById('disease').textContent = 'Error fetching disease information';
            document.getElementById('description').textContent = '';
        });
}

// Call the function when the DOM content is loaded
document.addEventListener('DOMContentLoaded', fetchDiseaseInformation);
