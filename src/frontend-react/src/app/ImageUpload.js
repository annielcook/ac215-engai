import React, { useState } from 'react';
import DataService from './services/DataService';

const ImageUpload = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const acceptableDataTypes = ['jpg', 'png', 'jpeg']

  const handleImageChange = (e) => {
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
        setSelectedImage(null)
        return
    }
    const formData = new FormData();
    formData.append('image', selectedImage);
    formData.append('file_type', fileType);

    DataService.Predict(formData)
        .then(function(response) {
            console.log(response);
        });
  };

  return (
    <div>
      <input type="file" accept="image/*" onChange={handleImageChange} />
      <button onClick={handleUpload} disabled={!selectedImage}>
        Upload
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
    </div>
  );
};

export default ImageUpload;
