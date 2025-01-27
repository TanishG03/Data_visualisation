
import React, { useState, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Text, Line } from '@react-three/drei';

const HeidiFinal = () => {
    const [data, setData] = useState(null);
    const [hoveredRegion, setHoveredRegion] = useState(null);
    const [hoveredPoints, setHoveredPoints] = useState([]);
    const [hoveredMatrixType, setHoveredMatrixType] = useState(null);
  
    useEffect(() => {
      fetch('/visualization_data_final_Dense_Concentric_Spheres.json')
      // fetch('/visualization_data_final_Dense_Toroidal_Structure.json')
    // visualization_data_final_Dense_3D_Spiral
    // fetch('/visualization_data_final_Dense_3D_Spiral.json')
    // fetch('/visualization_data_final_Interleaved_Helical_Structures.json')

        .then((response) => response.json())
        .then((jsonData) => {
          setData(jsonData);
        })
        .catch((error) => console.error('Error loading data:', error));
    }, []);
  
    const handleMatrixHover = (rowIndex, colIndex, matrixType) => {
      if (data) {
        const matrixValue = matrixType === 'ordered' 
          ? data.heidi_matrix.ordered[rowIndex][colIndex]
          : data.heidi_matrix.unordered[rowIndex][colIndex];
  
        if (matrixValue >= 0) {
          if (matrixType === 'ordered') {
            // Map ordered indices to original points
            const originalRowIndex = data.ordering[rowIndex];
            const originalColIndex = data.ordering[colIndex];
            setHoveredPoints([originalRowIndex, originalColIndex]);
          } else {
            // Use direct indices for unordered matrix
            setHoveredPoints([rowIndex, colIndex]);
          }
        } else {
          setHoveredPoints([]);
        }
        setHoveredRegion([rowIndex, colIndex]);
        setHoveredMatrixType(matrixType);
      }
    };

    const clearHover = () => {
        setHoveredPoints([]);
        setHoveredRegion(null);
        setHoveredMatrixType(null);
      };
    
      const getColorFromValue = (value, maxValue) => {
        const hue = 240 - (value / maxValue) * 240;
        return `hsl(${hue}, 100%, 50%)`;
      };

  const Axes = () => {
    return (
      <>
        {/* X-axis */}
        <Line 
          points={[[0, 0, 0], [10, 0, 0]]}
          color="red"
          lineWidth={2}
        />
        <Text 
          position={[11, 0, 0]}
          color="red"
          fontSize={0.5}
        >
          X
        </Text>

        {/* Y-axis */}
        <Line 
          points={[[0, 0, 0], [0, 10, 0]]}
          color="green"
          lineWidth={2}
        />
        <Text 
          position={[0, 11, 0]}
          color="green"
          fontSize={0.5}
        >
          Y
        </Text>

        {/* Z-axis */}
        <Line 
          points={[[0, 0, 0], [0, 0, 10]]}
          color="blue"
          lineWidth={2}
        />
        <Text 
          position={[0, 0, 11]}
          color="blue"
          fontSize={0.5}
        >
          Z
        </Text>
      </>
    );
  };

  const PointCloud = () => {
    if (!data) return null;

    return (
      <>
        {data.points.map((point, index) => {
          const isHighlighted = hoveredPoints.includes(index);
          return (
            <group key={index}>
              <mesh
                position={[point.x, point.y, point.z]}
                onPointerOver={() => setHoveredPoints([index])}
                onPointerOut={clearHover}
              >
                <sphereGeometry args={[isHighlighted ? 0.1 : 0.05, 32, 32]} />
                <meshStandardMaterial
                  color={isHighlighted ? '#ff0000' : '#8884d8'}
                  emissive={isHighlighted ? '#ff0000' : '#000000'}
                />
              </mesh>
              {isHighlighted && (
                <>
                  <Text
                    position={[point.x + 0.5, point.y + 0.5, point.z + 0.5]}
                    fontSize={0.5}
                    color="#ff0000"
                  >
                    Point {index}
                  </Text>
                  <Line
                    points={[
                      [point.x + 0.3, point.y + 0.3, point.z + 0.3],
                      [point.x, point.y, point.z]
                    ]}
                    color="#ff0000"
                    lineWidth={2}
                  />
                </>
              )}
            </group>
          );
        })}
      </>
    );
  };

  const getMatrixDisplay = (matrixData, type, maxValue) => {
    const matrixSize = matrixData.length;
    const cellSize = 350 / matrixSize;

    return (
      <div style={{ width: '400px', height: '450px', border: '1px solid #ccc', padding: '10px', position: 'relative' }}>
        <h3 style={{ fontSize: '18px', marginBottom: '10px' }}>
          {type === 'ordered' ? 'Ordered HEIDI Matrix' : 'Unordered HEIDI Matrix'}
        </h3>
        <svg width={370} height={370}>
          <g transform="translate(10, 10)">
            {matrixData.map((row, i) => (
              <g key={`row-${i}`}>
                {row.map((value, j) => (
                  <rect
                    key={`cell-${i}-${j}`}
                    x={j * cellSize}
                    y={i * cellSize}
                    width={cellSize}
                    height={cellSize}
                    fill={getColorFromValue(value, maxValue)}
                    opacity={
                      hoveredRegion &&
                      hoveredMatrixType === type &&
                      (hoveredRegion[0] === i || hoveredRegion[1] === j)
                        ? 1
                        : 0.7
                    }
                    stroke={
                      hoveredRegion &&
                      hoveredMatrixType === type &&
                      hoveredRegion[0] === i &&
                      hoveredRegion[1] === j
                        ? '#ff0000'
                        : 'none'
                    }
                    strokeWidth={2}
                    onMouseEnter={() => handleMatrixHover(i, j, type)}
                    onMouseLeave={clearHover}
                  />
                ))}
              </g>
            ))}
          </g>
        </svg>
        
        {hoveredRegion && 
         hoveredMatrixType === type && 
         matrixData[hoveredRegion[0]][hoveredRegion[1]] > 0 && (
          <div style={{
            position: 'absolute',
            bottom: '10px',
            left: '50%',
            transform: 'translateX(-50%)',
            background: 'rgba(255, 255, 255, 0.9)',
            padding: '10px',
            border: '1px solid #ccc',
            borderRadius: '5px',
            boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
          }}>
            <p style={{ margin: '5px 0' }}>Matrix Position: {hoveredRegion[0]} ↔ {hoveredRegion[1]}</p>
            {type === 'ordered' && (
              <p style={{ margin: '5px 0' }}>Original Points: {data.ordering[hoveredRegion[0]]} ↔ {data.ordering[hoveredRegion[1]]}</p>
            )}
            <p style={{ margin: '5px 0' }}>Value: {matrixData[hoveredRegion[0]][hoveredRegion[1]].toFixed(3)}</p>
          </div>
        )}
      </div>
    );
  };

  if (!data) return <div>Loading data...</div>;

  const maxMatrixValue = Math.max(
    ...data.heidi_matrix.ordered.flat(),
    ...data.heidi_matrix.unordered.flat()
  );

  return (
    <div style={{ padding: '20px', textAlign: 'center', fontFamily: 'Arial, sans-serif' }}>
      <h2 style={{ fontSize: '24px', marginBottom: '20px' }}>3D Clusters and HEIDI Matrix Analysis</h2>

      <div style={{ display: 'flex', justifyContent: 'center', gap: '20px', flexWrap: 'wrap' }}>
        <div style={{ width: '500px', height: '500px', border: '1px solid #ccc', padding: '10px' }}>
          <h3 style={{ fontSize: '18px', marginBottom: '10px' }}>3D Point Cloud</h3>
          <Canvas
            camera={{ position: [15, 15, 15], fov: 50 }}
            style={{ width: '100%', height: '100%' }}
          >
            <ambientLight intensity={0.5} />
            <pointLight position={[10, 10, 10]} />
            <Axes />
            <PointCloud />
            <OrbitControls enableRotate={true} enableZoom={true} enablePan={true} />
          </Canvas>
        </div>

        <div style={{ display: 'flex', gap: '20px', flexDirection: 'row' }}>
          {getMatrixDisplay(data.heidi_matrix.unordered, 'unordered', maxMatrixValue)}
          {getMatrixDisplay(data.heidi_matrix.ordered, 'ordered', maxMatrixValue)}
        </div>
      </div>

      <div style={{
        marginTop: '20px',
        color: '#555',
        maxWidth: '800px',
        margin: '0 auto',
        textAlign: 'center',
      }}>
        <p>Hover over cells in either HEIDI matrix to highlight and label the corresponding points in the scatter plot.</p>
        <p>Color intensity represents the strength of connections between points.</p>
        <p>The ordered matrix shows points reordered by KNN relationships, while the unordered matrix shows the original point ordering.</p>
      </div>
    </div>
  );
};

export default HeidiFinal;