// src/components/D3Visualization.js
import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';

function D3Visualization({ data }) {
  const svgRef = useRef();

  useEffect(() => {
    if (!data) return;

    const width = 500;
    const height = 500;
    const svg = d3.select(svgRef.current)
                  .attr("width", width)
                  .attr("height", height)
                  .style("margin", "10px");

    const numRows = data.length;
    const numCols = data[0].length;

    const cellWidth = width / numCols;
    const cellHeight = height / numRows;

    const colorScale = d3.scaleLinear()
                        .domain([0, 1])
                        .range(["white", "black"]);

    svg.selectAll("rect")
       .data(data.flat())
       .enter()
       .append("rect")
       .attr("x", (d, i) => (i % numCols) * cellWidth)
       .attr("y", (d, i) => Math.floor(i / numCols) * cellHeight)
       .attr("width", cellWidth)
       .attr("height", cellHeight)
       .attr("fill", d => colorScale(d[0]));  // Use only one channel (e.g., red)

  }, [data]);

  return <svg ref={svgRef}></svg>;
}

export default D3Visualization;
