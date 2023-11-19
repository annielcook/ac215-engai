import React, { useState, useRef } from 'react';
import DataService from '../../services/DataService';
import ModelToggle from '../ModelToggle/ModelToggle';
import BreedParse from '../../services/BreedParse';
import './ImageUpload.css'; // Import the CSS file

const ImageUpload = ({ onPredictionChange }) => {
    // Image file components
    const [selectedImage, setSelectedImage] = useState(null);
    const inputFile = useRef(null);
    const acceptableDataTypes = ['jpg', 'png', 'jpeg']
    // Model state
    const [selectedModel, setSelectedModel] = useState('hosted-model');
    // Prediction results with breed and probability
    const [predictedBreed, setPredictedBreed] = useState(null);
    const [predictedProb, setPredictedProb] = useState(null);
    // Boolean for showing the refresh button
    const [hasPreview, setHasPreview] = useState(false);
    const [showRefresh, setShowRefresh] = useState(false);

    const handleModelChange = (event) => {
      setSelectedModel(event.target.value);
      // setSelectedModel((prevModel) => event.target.value);
    };

    const handleImageChange = (e) => {
      inputFile.current.click();
      const file = e.target.files[0];

      if (file) {
        // You can perform additional checks or validation here
        setSelectedImage(file);
        setHasPreview(true);
        setShowRefresh(false);
        setPredictedBreed(null);
        setPredictedProb(null);
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
      // Use the local model if set, default to remote model.
      const useLocalModel = selectedModel === 'local-model';
      const formData = new FormData();
      formData.append('image', selectedImage);
      formData.append('file_type', fileType);
      formData.append('use_local_model', useLocalModel);

      DataService.Predict(formData)
          .then(function(response) {
            console.log(response.data.predicted_breed);
            console.log(response.data.max_probability);
            if (response.data.max_probability < 0.25) {
              setPredictedBreed("Confidence too low to predict");
            } else {
              const predBreed = BreedParse(response.data.predicted_breed);
              setPredictedBreed(predBreed);
              const probWithTwoDec = Number((parseFloat(response.data.max_probability)*100).toFixed(2));
              const percProb = String(probWithTwoDec) + "%";
              setPredictedProb(percProb);
              onPredictionChange(predBreed);
            }
            setHasPreview(false);
            setShowRefresh(true);
          });
    };

    const clearImage = () => {
      if (inputFile.current) {
        inputFile.current.value = '';
        setSelectedImage(null);
        setPredictedBreed(null);
        setPredictedProb(null);
        setShowRefresh(false);
      }
    };

    return (
      <div style={{alignItems: 'center'}}>
        <div className="upload-container">
          < ModelToggle selectedModel={selectedModel} onModelChange={handleModelChange} />
        </div>
        <br/>
        <div className="upload-container">
          <label htmlFor="file-input" className="site-button">Choose File</label>
          <input id="file-input"
            className="custom-file-upload-input"
            type="file"
            accept="image/*"
            capture="camera"
            ref={inputFile}
            onChange={handleImageChange}
          />
        </div>
        {selectedImage && (
          <div className="upload-container">
              <h2>Preview:</h2>
            <img className="uploaded-img" src={URL.createObjectURL(selectedImage)} alt="Selected" />
          </div>
        )}
        <br/>
        <div className="upload-container">
          {hasPreview && (
            <button
              className="site-button"
              onClick={handleUpload}>
              Upload
            </button>
          )}
          {showRefresh && (
            <button
              className="site-button"
              onClick={clearImage}>
              Clear Results
            </button>
          )}
        </div>
        <div className="upload-container">
          <table style={{ borderCollapse: 'collapse', width: '300px', marginTop: '10px' }}>
            <tbody>
              {predictedBreed &&
              (<tr>
                <td style={{ padding: '8px' }}><strong>Predicted Breed</strong></td>
                <td style={{ padding: '8px' }}>{predictedBreed}</td>
              </tr>
              )}
              {predictedProb &&
              (<tr>
                <td style={{ padding: '8px' }}><strong>Confidence Level</strong></td>
                <td style={{ padding: '8px' }}>{predictedProb}</td>
              </tr>
              )}
            </tbody>
          </table>
          {/* {predictedBreed && (<p><strong>Predicted Breed:</strong> {predictedBreed}</p>)} */}
          {/* {predictedProb && (<p><strong>Confidence Level:</strong> {predictedProb}</p>)} */}
        </div>
      </div>
    );
};

export default ImageUpload;
