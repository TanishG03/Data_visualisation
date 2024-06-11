import React, { useState } from 'react';
import axios from 'axios';

const UploadCSV = () => {
  const [file, setFile] = useState(null);
  const [option, setOption] = useState('1');
  const [outputData, setOutputData] = useState(null);
  const [outputImages, setOutputImages] = useState(null);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleOptionChange = (e) => {
    setOption(e.target.value);
  };

  const handleUpload = async () => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('option', option);

    try {
      const response = await axios.post('http://localhost:5000/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      console.log('File uploaded successfully');
      console.log('Output data:', response.data.data); // Output data received from the backend
      console.log('Output images:', response.data.images); // Output images received from the backend
      setOutputData(response.data.data); // Set the received data in state for rendering
      setOutputImages(response.data.images); // Set the received images in state for rendering
    } catch (error) {
      console.error('Error uploading file: ', error);
    }
  };

  return (
    <div>
      <input type="file" onChange={handleFileChange} />
      <select value={option} onChange={handleOptionChange}>
        <option value="1">Top KNN</option>
        <option value="2">Top Spiral</option>
        <option value="3">Single Dim KNN</option>
        <option value="4">Option 4</option>
      </select>
      <button onClick={handleUpload}>Upload</button>

      {/* Render the output data */}
      {outputData && (
        <div>
          <h2>Processed Data</h2>
          <pre>{JSON.stringify(outputData, null, 2)}</pre>
        </div>
      )}

      {/* Render the output images */}
      {outputImages && Object.keys(outputImages).map((imageName) => (
        <div key={imageName}>
          <h2>{imageName}</h2>
          <img src={`data:image/jpeg;base64,${outputImages[imageName]}`} alt={imageName} />
        </div>
      ))}
    </div>
  );
};

export default UploadCSV;
