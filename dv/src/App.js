import React from 'react';
import UploadCSV from './components/UploadCSV';
import HeidiVisualization from './components/HeidiVisualization';
import HeidiFinal from './components/HeidiFinal';

function App() {
  return (
    <div>
      <h1>CSV Upload and Visualization</h1>
      <UploadCSV />
      {/* <HeidiVisualization /> */}
      <HeidiFinal />
    </div>
  );
}

export default App;
