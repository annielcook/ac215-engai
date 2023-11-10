const axios = require('axios');
const API_SERVICE_BASE_URL = process.env.API_SERVICE_BASE_URL

const DataService = {
    // Predict sends a POST request to the api-service to receive a prediction result.
    Predict: async function(formData) {
        return await axios.post(API_SERVICE_BASE_URL + '/predict', formData, {
            headers: {
                'Content-Type': 'multipart/form-data'
            }
        });
    }
}

export default DataService;