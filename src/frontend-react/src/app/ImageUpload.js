import React, { useState, useRef } from 'react';
import DataService from './services/DataService';
// import Prediction from './Prediction';

const ImageUpload = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const acceptableDataTypes = ['jpg', 'png', 'jpeg']
  const [prediction, setPrediction] = useState(null);
  const inputFile = useRef(null);

  const handleImageChange = (e) => {
    inputFile.current.click();
    const file = e.target.files[0];

    if (file) {
      // You can perform additional checks or validation here
      setSelectedImage(file);
    }
  };

  const handleUpload = () => {
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

    DataService.Predict(formData)
        .then(function(response) {
          console.log(response.data.predicted_breed);
          console.log(response.data.max_probability);
          setPrediction(response.data);
        });
  };

  const clearImage = () => {
    if (inputFile.current) {
      inputFile.current.value = '';
      setPrediction(null);
      setSelectedImage(null);
    }
  };

  return (
    <div>
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
      {prediction && (
        <div>
          <p>Predicted Breed: {prediction.predicted_breed}</p>
          <p>Confidence: {prediction.max_probability}</p>
        </div>
      )}
    </div>
  );
};

export default ImageUpload;
