

// import React, { useState, useEffect } from 'react';
// import { Canvas } from '@react-three/fiber';
// import { OrbitControls, Text, Line } from '@react-three/drei';

// const HeidiVisualization = () => {
//   const [data, setData] = useState(null);
//   const [hoveredRegion, setHoveredRegion] = useState(null);
//   const [hoveredPoints, setHoveredPoints] = useState([]);

//   useEffect(() => {
//     // fetch('/visualization_data_Dense_Concentric_Spheres.json')
//   //  fetch('/visualization_data_Dense_Toroidal_Structure.json')
//   // fetch('/visualization_data_Dense_3D_Spiral.json')
//   // fetch('/visualization_data_unordered_Dense_Toroidal_Structure.json')
//   // fetch('visualization_data_unordered_Dense_Concentric_Spheres.json')

//   fetch('/visualization_data_Interleaved_Helical_Structures.json')

//       .then((response) => response.json())
//       .then((jsonData) => {
//         setData(jsonData);
//       })
//       .catch((error) => console.error('Error loading data:', error));
//   }, []);

//   // const handleMatrixHover = (rowIndex, colIndex) => {
//   //   if (data && data.heidi_matrix[rowIndex][colIndex] >= 0) {
//   //     setHoveredPoints([rowIndex, colIndex]);
//   //   } else {
//   //     setHoveredPoints([]);
//   //   }
//   //   setHoveredRegion([rowIndex, colIndex]);
//   // };

//   const handleMatrixHover = (rowIndex, colIndex) => {
//     if (data && data.heidi_matrix[rowIndex][colIndex] >= 0) {
//       // Map the ordered indices back to original indices
//       const originalRowIndex = data.ordering[rowIndex];
//       const originalColIndex = data.ordering[colIndex];
//       setHoveredPoints([originalRowIndex, originalColIndex]);
//     } else {
//       setHoveredPoints([]);
//     }
//     setHoveredRegion([rowIndex, colIndex]);
// };

//   const clearHover = () => {
//     setHoveredPoints([]);
//     setHoveredRegion(null);
//   };

//   const getColorFromValue = (value, maxValue) => {
//     const hue = 240 - (value / maxValue) * 240;
//     return `hsl(${hue}, 100%, 50%)`;
//   };

//   const Axes = () => {
//     return (
//       <>
//         {/* X-axis */}
//         <Line 
//           points={[[0, 0, 0], [10, 0, 0]]}
//           color="red"
//           lineWidth={2}
//         />
//         <Text 
//           position={[11, 0, 0]}
//           color="red"
//           fontSize={0.5}
//         >
//           X
//         </Text>

//         {/* Y-axis */}
//         <Line 
//           points={[[0, 0, 0], [0, 10, 0]]}
//           color="green"
//           lineWidth={2}
//         />
//         <Text 
//           position={[0, 11, 0]}
//           color="green"
//           fontSize={0.5}
//         >
//           Y
//         </Text>

//         {/* Z-axis */}
//         <Line 
//           points={[[0, 0, 0], [0, 0, 10]]}
//           color="blue"
//           lineWidth={2}
//         />
//         <Text 
//           position={[0, 0, 11]}
//           color="blue"
//           fontSize={0.5}
//         >
//           Z
//         </Text>
//       </>
//     );
//   };

//   const PointCloud = () => {
//     if (!data) return null;

//     return (
//       <>
//         {data.points.map((point, index) => {
//           const isHighlighted = hoveredPoints.includes(index);
//           return (
//             <group key={index}>
//               <mesh
//                 position={[point.x, point.y, point.z]}
//                 onPointerOver={() => setHoveredPoints([index])}
//                 onPointerOut={clearHover}
//               >
//                 <sphereGeometry args={[isHighlighted ? 0.1 : 0.05, 32, 32]} />
//                 <meshStandardMaterial
//                   color={isHighlighted ? '#ff0000' : '#8884d8'}
//                   emissive={isHighlighted ? '#ff0000' : '#000000'}
//                 />
//               </mesh>
//               {isHighlighted && (
//                 <>
//                   {/* Point label */}
//                   <Text
//                     position={[point.x + 0.5, point.y + 0.5, point.z + 0.5]}
//                     fontSize={0.5}
//                     color="#ff0000"
//                   >
//                     Point {index}
//                   </Text>
//                   {/* Arrow to point */}
//                   <Line
//                     points={[
//                       [point.x + 0.3, point.y + 0.3, point.z + 0.3],
//                       [point.x, point.y, point.z]
//                     ]}
//                     color="#ff0000"
//                     lineWidth={2}
//                   />
//                 </>
//               )}
//             </group>
//           );
//         })}
//       </>
//     );
//   };

//   if (!data) return <div>Loading data...</div>;

//   const matrixSize = data.heidi_matrix.length;
//   const cellSize = 400 / matrixSize;
//   const maxMatrixValue = Math.max(...data.heidi_matrix.flat());

//   return (
//     <div style={{ padding: '20px', textAlign: 'center', fontFamily: 'Arial, sans-serif' }}>
//       <h2 style={{ fontSize: '24px', marginBottom: '20px' }}>3D Clusters and HEIDI Matrix Analysis</h2>

//       <div style={{ display: 'flex', justifyContent: 'center', gap: '20px', flexWrap: 'wrap' }}>
//         {/* 3D Scatter Plot */}
//         <div style={{ width: '500px', height: '500px', border: '1px solid #ccc', padding: '10px' }}>
//           <h3 style={{ fontSize: '18px', marginBottom: '10px' }}>3D Point Cloud</h3>
//           <Canvas
//             camera={{ position: [15, 15, 15], fov: 50 }}
//             style={{ width: '100%', height: '100%' }}
//           >
//             <ambientLight intensity={0.5} />
//             <pointLight position={[10, 10, 10]} />
//             <Axes />
//             <PointCloud />
//             <OrbitControls enableRotate={false} enableZoom={true} enablePan={false} />
//           </Canvas>
//         </div>

//         {/* HEIDI Matrix */}
//         <div style={{ width: '500px', height: '500px', border: '1px solid #ccc', padding: '10px', position: 'relative' }}>
//           <h3 style={{ fontSize: '18px', marginBottom: '10px' }}>HEIDI Matrix</h3>
//           <svg width={450} height={450}>
//             <g transform="translate(20, 20)">
//               {data.heidi_matrix.map((row, i) => (
//                 <g key={`row-${i}`}>
//                   {row.map((value, j) => (
//                     <rect
//                       key={`cell-${i}-${j}`}
//                       x={j * cellSize}
//                       y={i * cellSize}
//                       width={cellSize}
//                       height={cellSize}
//                       fill={getColorFromValue(value, maxMatrixValue)}
//                       opacity={
//                         hoveredRegion &&
//                         (hoveredRegion[0] === i || hoveredRegion[1] === j)
//                           ? 1
//                           : 0.7
//                       }
//                       stroke={
//                         hoveredRegion &&
//                         hoveredRegion[0] === i &&
//                         hoveredRegion[1] === j
//                           ? '#ff0000'
//                           : 'none'
//                       }
//                       strokeWidth={2}
//                       onMouseEnter={() => handleMatrixHover(i, j)}
//                       onMouseLeave={clearHover}
//                     />
//                   ))}
//                 </g>
//               ))}
//             </g>
//           </svg>
          
//           {hoveredRegion && data.heidi_matrix[hoveredRegion[0]][hoveredRegion[1]] > 0 && (
//             <div style={{
//               position: 'absolute',
//               top: '50%',
//               left: '50%',
//               transform: 'translate(-50%, -50%)',
//               background: 'rgba(255, 255, 255, 0.9)',
//               padding: '10px',
//               border: '1px solid #ccc',
//               borderRadius: '5px',
//               boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
//             }}>
//               <p>Points: {hoveredRegion[0]} ↔ {hoveredRegion[1]}</p>
//               <p>Value: {data.heidi_matrix[hoveredRegion[0]][hoveredRegion[1]].toFixed(3)}</p>
              
//             </div>
//           )}
//         </div>
//       </div>

//       <div style={{
//         marginTop: '20px',
//         color: '#555',
//         maxWidth: '600px',
//         margin: '0 auto',
//         textAlign: 'center',
//       }}>
//         <p>Hover over a cell in the HEIDI matrix to highlight and label the corresponding points in the scatter plot.</p>
//         <p>Color intensity in the matrix represents the strength of connections between points.</p>
//       </div>
//     </div>
//   );
// };

// export default HeidiVisualization;



import React, { useState, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Text, Line } from '@react-three/drei';

const HeidiVisualization = () => {
  const [data, setData] = useState(null);
  const [hoveredRegion, setHoveredRegion] = useState(null);
  const [hoveredPoints, setHoveredPoints] = useState([]);

  useEffect(() => {
    // fetch('/visualization_data_Dense_Concentric_Spheres.json')
  //  fetch('/visualization_data_Dense_Toroidal_Structure.json')
  // fetch('/visualization_data_Dense_3D_Spiral.json')
  // fetch('/visualization_data_unordered_Dense_Toroidal_Structure.json')
  fetch('visualization_data_ordered_Dense_Concentric_Spheres.json')

  // fetch('/visualization_data_Interleaved_Helical_Structures.json')

      .then((response) => response.json())
      .then((jsonData) => {
        setData(jsonData);
      })
      .catch((error) => console.error('Error loading data:', error));
  }, []);

  // const handleMatrixHover = (rowIndex, colIndex) => {
  //   if (data && data.heidi_matrix[rowIndex][colIndex] >= 0) {
  //     setHoveredPoints([rowIndex, colIndex]);
  //   } else {
  //     setHoveredPoints([]);
  //   }
  //   setHoveredRegion([rowIndex, colIndex]);
  // };

  const handleMatrixHover = (rowIndex, colIndex) => {
    if (data && data.heidi_matrix[rowIndex][colIndex] >= 0) {
      // Map the ordered indices back to original indices
      const originalRowIndex = data.ordering[rowIndex];
      const originalColIndex = data.ordering[colIndex];
      setHoveredPoints([originalRowIndex, originalColIndex]);
    } else {
      setHoveredPoints([]);
    }
    setHoveredRegion([rowIndex, colIndex]);
};

  const clearHover = () => {
    setHoveredPoints([]);
    setHoveredRegion(null);
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
                  {/* Point label */}
                  <Text
                    position={[point.x + 0.5, point.y + 0.5, point.z + 0.5]}
                    fontSize={0.5}
                    color="#ff0000"
                  >
                    Point {index}
                  </Text>
                  {/* Arrow to point */}
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

  if (!data) return <div>Loading data...</div>;

  const matrixSize = data.heidi_matrix.length;
  const cellSize = 400 / matrixSize;
  const maxMatrixValue = Math.max(...data.heidi_matrix.flat());

  return (
    <div style={{ padding: '20px', textAlign: 'center', fontFamily: 'Arial, sans-serif' }}>
      <h2 style={{ fontSize: '24px', marginBottom: '20px' }}>3D Clusters and HEIDI Matrix Analysis</h2>

      <div style={{ display: 'flex', justifyContent: 'center', gap: '20px', flexWrap: 'wrap' }}>
        {/* 3D Scatter Plot */}
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
            <OrbitControls enableRotate={false} enableZoom={true} enablePan={false} />
          </Canvas>
        </div>

        {/* HEIDI Matrix */}
        <div style={{ width: '500px', height: '500px', border: '1px solid #ccc', padding: '10px', position: 'relative' }}>
          <h3 style={{ fontSize: '18px', marginBottom: '10px' }}>HEIDI Matrix</h3>
          <svg width={450} height={450}>
            <g transform="translate(20, 20)">
              {data.heidi_matrix.map((row, i) => (
                <g key={`row-${i}`}>
                  {row.map((value, j) => (
                    <rect
                      key={`cell-${i}-${j}`}
                      x={j * cellSize}
                      y={i * cellSize}
                      width={cellSize}
                      height={cellSize}
                      fill={getColorFromValue(value, maxMatrixValue)}
                      opacity={
                        hoveredRegion &&
                        (hoveredRegion[0] === i || hoveredRegion[1] === j)
                          ? 1
                          : 0.7
                      }
                      stroke={
                        hoveredRegion &&
                        hoveredRegion[0] === i &&
                        hoveredRegion[1] === j
                          ? '#ff0000'
                          : 'none'
                      }
                      strokeWidth={2}
                      onMouseEnter={() => handleMatrixHover(i, j)}
                      onMouseLeave={clearHover}
                    />
                  ))}
                </g>
              ))}
            </g>
          </svg>
          
          {hoveredRegion && data.heidi_matrix[hoveredRegion[0]][hoveredRegion[1]] > 0 && (
            <div style={{
              position: 'absolute',
              top: '50%',
              left: '50%',
              transform: 'translate(-50%, -50%)',
              background: 'rgba(255, 255, 255, 0.9)',
              padding: '10px',
              border: '1px solid #ccc',
              borderRadius: '5px',
              boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
            }}>
<p>Ordered Points: {hoveredRegion[0]} ↔ {hoveredRegion[1]}</p>
    <p>Original Points: {data.ordering[hoveredRegion[0]]} ↔ {data.ordering[hoveredRegion[1]]}</p>
    <p>Value: {data.heidi_matrix[hoveredRegion[0]][hoveredRegion[1]].toFixed(3)}</p>
            </div>
          )}
        </div>
      </div>

      <div style={{
        marginTop: '20px',
        color: '#555',
        maxWidth: '600px',
        margin: '0 auto',
        textAlign: 'center',
      }}>
        <p>Hover over a cell in the HEIDI matrix to highlight and label the corresponding points in the scatter plot.</p>
        <p>Color intensity in the matrix represents the strength of connections between points.</p>
      </div>
    </div>
  );
};

export default HeidiVisualization;