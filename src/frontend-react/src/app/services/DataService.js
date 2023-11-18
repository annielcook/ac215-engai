const axios = require('axios');
const API_SERVICE_BASE_URL = process.env.REACT_APP_API_SERVICE_BASE_URL

// const predict = async (formData) => {
//     return await axios.post(API_SERVICE_BASE_URL + '/predict', formData, {
//         headers: {
//             'Content-Type': 'multipart/form-data'
//         }
//     });
//   };
  

const DataService = {
    // Predict sends a POST request to the api-service to receive a prediction result.
    Predict: async function(formData, selectedModel) {
        return await axios.post(API_SERVICE_BASE_URL + '/predict', formData, {
            headers: {
                'Content-Type': 'multipart/form-data'
            }
        });
    }
}

export default DataService;