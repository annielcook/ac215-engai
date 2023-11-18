import React, { useState, useRef } from 'react';
import DataService from '../services/DataService';
import ModelToggle from './ModelToggle';
// import Prediction from './Prediction';

const ImageUpload = () => {
    const [selectedImage, setSelectedImage] = useState(null);
    const [selectedModel, setSelectedModel] = useState('hosted-model');
    const acceptableDataTypes = ['jpg', 'png', 'jpeg']
    const [predictedBreed, setPredictedBreed] = useState(null);
    const [predictedProb, setPredictedProb] = useState(null);
    const inputFile = useRef(null);
    // const percProb = null;

    const handleModelChange = (event) => {
      // setSelectedModel(event.target.value);
      setSelectedModel((prevModel) => event.target.value);
      // console.log(selectedModel);
    };
    const handleImageChange = (e) => {
      inputFile.current.click();
      const file = e.target.files[0];

      if (file) {
        // You can perform additional checks or validation here
        setSelectedImage(file);
      }
    };

    const handleUpload = () => {
      console.log(selectedModel);
      console.log('Selected Image:', selectedImage);
      const fileType = selectedImage.name.split('.')[1]
      // Validate that the file type is valid.
      if (!acceptableDataTypes.includes(fileType)) {
          alert('Only supports jpg or png file!')
          // Reset selectedImage
          setSelectedImage(null);
          return;
      }
      const formData = new FormData();
      formData.append('image', selectedImage);
      formData.append('file_type', fileType);

      DataService.Predict(formData, selectedModel)
          .then(function(response) {
            console.log(response.data.predicted_breed);
            console.log(response.data.max_probability);
            setPredictedBreed(response.data.predicted_breed);
            const probWithTwoDec = Number((parseFloat(response.data.max_probability)*100).toFixed(2));
            const percProb = String(probWithTwoDec) + "%";
            setPredictedProb(percProb);
          });
    };

    const clearImage = () => {
      if (inputFile.current) {
        inputFile.current.value = '';
        setSelectedImage(null);
        setPredictedBreed(null);
        setPredictedProb(null);
      }
    };

    return (
      <div>
        <ModelToggle selectedModel={selectedModel} onModelChange={handleModelChange} />
        <br />
        <input
          type="file"
          accept="image/*"
          ref={inputFile}
          onChange={handleImageChange}
        />
        <button onClick={handleUpload} disabled={!selectedImage}>
          Upload
        </button>
        <button onClick={clearImage} disabled={!selectedImage}>
          Refresh
        </button>
        {selectedImage && (
          <div>
            <h2>Preview:</h2>
            <img
              src={URL.createObjectURL(selectedImage)}
              alt="Selected"
              style={{ maxWidth: '100%', maxHeight: '300px' }}
            />
          </div>
        )}
        {/* {prediction && ( */}
          <div>
            {predictedBreed && (<p>Predicted Breed: {predictedBreed}</p>)}
            {predictedProb && (<p>Confidence: {predictedProb}</p>)}
          </div>
        {/* )} */}
      </div>
    );
};

export default ImageUpload;
