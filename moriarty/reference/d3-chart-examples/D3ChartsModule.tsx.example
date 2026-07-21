// ============================================
// 1. AREA CHART
// ============================================
// File: AreaChart.tsx
import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';

interface AreaChartProps {
  data: { date: Date; value: number }[];
  width?: number;
  height?: number;
  margin?: { top: number; right: number; bottom: number; left: number };
  color?: string;
}

export const AreaChart: React.FC<AreaChartProps> = ({
  data,
  width = 800,
  height = 400,
  margin = { top: 20, right: 30, bottom: 30, left: 50 },
  color = '#4299e1'
}) => {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || !data.length) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const g = svg
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Scales
    const xScale = d3
      .scaleTime()
      .domain(d3.extent(data, d => d.date) as [Date, Date])
      .range([0, innerWidth]);

    const yScale = d3
      .scaleLinear()
      .domain([0, d3.max(data, d => d.value) || 0])
      .range([innerHeight, 0]);

    // Area generator
    const area = d3
      .area<{ date: Date; value: number }>()
      .x(d => xScale(d.date))
      .y0(innerHeight)
      .y1(d => yScale(d.value))
      .curve(d3.curveMonotoneX);

    // Draw area
    g.append('path')
      .datum(data)
      .attr('fill', color)
      .attr('opacity', 0.7)
      .attr('d', area);

    // Line on top
    const line = d3
      .line<{ date: Date; value: number }>()
      .x(d => xScale(d.date))
      .y(d => yScale(d.value))
      .curve(d3.curveMonotoneX);

    g.append('path')
      .datum(data)
      .attr('fill', 'none')
      .attr('stroke', color)
      .attr('stroke-width', 2)
      .attr('d', line);

    // Axes
    g.append('g')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(d3.axisBottom(xScale));

    g.append('g').call(d3.axisLeft(yScale));
  }, [data, width, height, margin, color]);

  return <svg ref={svgRef} width={width} height={height} />;
};

// ============================================
// 2. BAR CHART
// ============================================
// File: BarChart.tsx
interface BarChartProps {
  data: { label: string; value: number }[];
  width?: number;
  height?: number;
  margin?: { top: number; right: number; bottom: number; left: number };
  color?: string;
  horizontal?: boolean;
}

export const BarChart: React.FC<BarChartProps> = ({
  data,
  width = 800,
  height = 400,
  margin = { top: 20, right: 30, bottom: 60, left: 50 },
  color = '#48bb78',
  horizontal = false
}) => {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || !data.length) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const g = svg
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    if (horizontal) {
      // Horizontal bar chart
      const xScale = d3
        .scaleLinear()
        .domain([0, d3.max(data, d => d.value) || 0])
        .range([0, innerWidth]);

      const yScale = d3
        .scaleBand()
        .domain(data.map(d => d.label))
        .range([0, innerHeight])
        .padding(0.2);

      g.selectAll('rect')
        .data(data)
        .enter()
        .append('rect')
        .attr('y', d => yScale(d.label) || 0)
        .attr('height', yScale.bandwidth())
        .attr('x', 0)
        .attr('width', d => xScale(d.value))
        .attr('fill', color);

      g.append('g')
        .attr('transform', `translate(0,${innerHeight})`)
        .call(d3.axisBottom(xScale));

      g.append('g').call(d3.axisLeft(yScale));
    } else {
      // Vertical bar chart
      const xScale = d3
        .scaleBand()
        .domain(data.map(d => d.label))
        .range([0, innerWidth])
        .padding(0.2);

      const yScale = d3
        .scaleLinear()
        .domain([0, d3.max(data, d => d.value) || 0])
        .range([innerHeight, 0]);

      g.selectAll('rect')
        .data(data)
        .enter()
        .append('rect')
        .attr('x', d => xScale(d.label) || 0)
        .attr('width', xScale.bandwidth())
        .attr('y', d => yScale(d.value))
        .attr('height', d => innerHeight - yScale(d.value))
        .attr('fill', color);

      g.append('g')
        .attr('transform', `translate(0,${innerHeight})`)
        .call(d3.axisBottom(xScale))
        .selectAll('text')
        .attr('transform', 'rotate(-45)')
        .style('text-anchor', 'end');

      g.append('g').call(d3.axisLeft(yScale));
    }
  }, [data, width, height, margin, color, horizontal]);

  return <svg ref={svgRef} width={width} height={height} />;
};

// ============================================
// 3. STREAM CHART (Stacked Area)
// ============================================
// File: StreamChart.tsx
interface StreamChartProps {
  data: { date: Date; [key: string]: Date | number }[];
  keys: string[];
  width?: number;
  height?: number;
  margin?: { top: number; right: number; bottom: number; left: number };
  colors?: string[];
}

export const StreamChart: React.FC<StreamChartProps> = ({
  data,
  keys,
  width = 800,
  height = 400,
  margin = { top: 20, right: 100, bottom: 30, left: 50 },
  colors = d3.schemeCategory10
}) => {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || !data.length) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const g = svg
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Stack generator
    const stack = d3.stack<any>()
      .keys(keys)
      .offset(d3.stackOffsetWiggle);

    const series = stack(data);

    // Scales
    const xScale = d3
      .scaleTime()
      .domain(d3.extent(data, d => d.date) as [Date, Date])
      .range([0, innerWidth]);

    const yScale = d3
      .scaleLinear()
      .domain([
        d3.min(series, s => d3.min(s, d => d[0])) || 0,
        d3.max(series, s => d3.max(s, d => d[1])) || 0
      ])
      .range([innerHeight, 0]);

    // Color scale
    const colorScale = d3
      .scaleOrdinal<string>()
      .domain(keys)
      .range(colors);

    // Area generator
    const area = d3
      .area<any>()
      .x(d => xScale(d.data.date))
      .y0(d => yScale(d[0]))
      .y1(d => yScale(d[1]))
      .curve(d3.curveBasis);

    // Draw streams
    g.selectAll('path')
      .data(series)
      .enter()
      .append('path')
      .attr('fill', d => colorScale(d.key))
      .attr('d', area)
      .attr('opacity', 0.8);

    // Axes
    g.append('g')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(d3.axisBottom(xScale));

    // Legend
    const legend = g
      .selectAll('.legend')
      .data(keys)
      .enter()
      .append('g')
      .attr('class', 'legend')
      .attr('transform', (d, i) => `translate(${innerWidth + 10},${i * 20})`);

    legend
      .append('rect')
      .attr('width', 18)
      .attr('height', 18)
      .attr('fill', d => colorScale(d));

    legend
      .append('text')
      .attr('x', 24)
      .attr('y', 9)
      .attr('dy', '.35em')
      .text(d => d);
  }, [data, keys, width, height, margin, colors]);

  return <svg ref={svgRef} width={width} height={height} />;
};

// ============================================
// 4. NETWORK DIAGRAM
// ============================================
// File: NetworkDiagram.tsx
interface Node {
  id: string;
  group?: number;
}

interface Link {
  source: string;
  target: string;
  value?: number;
}

interface NetworkDiagramProps {
  nodes: Node[];
  links: Link[];
  width?: number;
  height?: number;
  nodeRadius?: number;
}

export const NetworkDiagram: React.FC<NetworkDiagramProps> = ({
  nodes,
  links,
  width = 800,
  height = 600,
  nodeRadius = 8
}) => {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || !nodes.length) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const colorScale = d3.scaleOrdinal(d3.schemeCategory10);

    // Force simulation
    const simulation = d3
      .forceSimulation(nodes as any)
      .force(
        'link',
        d3.forceLink(links).id((d: any) => d.id)
      )
      .force('charge', d3.forceManyBody().strength(-200))
      .force('center', d3.forceCenter(width / 2, height / 2));

    // Links
    const link = svg
      .append('g')
      .selectAll('line')
      .data(links)
      .enter()
      .append('line')
      .attr('stroke', '#999')
      .attr('stroke-opacity', 0.6)
      .attr('stroke-width', d => Math.sqrt(d.value || 1));

    // Nodes
    const node = svg
      .append('g')
      .selectAll('circle')
      .data(nodes)
      .enter()
      .append('circle')
      .attr('r', nodeRadius)
      .attr('fill', d => colorScale(String(d.group || 0)))
      .call(
        d3
          .drag<any, any>()
          .on('start', (event, d: any) => {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
          })
          .on('drag', (event, d: any) => {
            d.fx = event.x;
            d.fy = event.y;
          })
          .on('end', (event, d: any) => {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
          })
      );

    // Labels
    const label = svg
      .append('g')
      .selectAll('text')
      .data(nodes)
      .enter()
      .append('text')
      .text(d => d.id)
      .attr('font-size', 10)
      .attr('dx', 12)
      .attr('dy', 4);

    simulation.on('tick', () => {
      link
        .attr('x1', (d: any) => d.source.x)
        .attr('y1', (d: any) => d.source.y)
        .attr('x2', (d: any) => d.target.x)
        .attr('y2', (d: any) => d.target.y);

      node.attr('cx', (d: any) => d.x).attr('cy', (d: any) => d.y);

      label.attr('x', (d: any) => d.x).attr('y', (d: any) => d.y);
    });

    return () => {
      simulation.stop();
    };
  }, [nodes, links, width, height, nodeRadius]);

  return <svg ref={svgRef} width={width} height={height} />;
};

// ============================================
// 5. CHORD DIAGRAM
// ============================================
// File: ChordDiagram.tsx
interface ChordDiagramProps {
  matrix: number[][];
  labels: string[];
  width?: number;
  height?: number;
  colors?: string[];
}

export const ChordDiagram: React.FC<ChordDiagramProps> = ({
  matrix,
  labels,
  width = 600,
  height = 600,
  colors = d3.schemeCategory10
}) => {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || !matrix.length) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const outerRadius = Math.min(width, height) * 0.5 - 40;
    const innerRadius = outerRadius - 30;

    const g = svg
      .append('g')
      .attr('transform', `translate(${width / 2},${height / 2})`);

    const chord = d3.chord().padAngle(0.05).sortSubgroups(d3.descending);

    const arc = d3.arc().innerRadius(innerRadius).outerRadius(outerRadius);

    const ribbon = d3.ribbon().radius(innerRadius);

    const colorScale = d3.scaleOrdinal().domain(d3.range(0, labels.length).map(String)).range(colors);

    const chords = chord(matrix);

    // Draw groups
    const group = g
      .append('g')
      .selectAll('g')
      .data(chords.groups)
      .enter()
      .append('g');

    group
      .append('path')
      .attr('fill', d => colorScale(String(d.index)))
      .attr('stroke', d => d3.rgb(colorScale(String(d.index)) as string).darker().toString())
      .attr('d', arc as any);

    group
      .append('text')
      .each(d => {
        (d as any).angle = (d.startAngle + d.endAngle) / 2;
      })
      .attr('dy', '.35em')
      .attr('transform', (d: any) => {
        return `
          rotate(${(d.angle * 180) / Math.PI - 90})
          translate(${outerRadius + 10})
          ${d.angle > Math.PI ? 'rotate(180)' : ''}
        `;
      })
      .attr('text-anchor', (d: any) => (d.angle > Math.PI ? 'end' : null))
      .text((d, i) => labels[i]);

    // Draw ribbons
    g.append('g')
      .attr('fill-opacity', 0.67)
      .selectAll('path')
      .data(chords)
      .enter()
      .append('path')
      .attr('d', ribbon as any)
      .attr('fill', d => colorScale(String(d.source.index)))
      .attr('stroke', d => d3.rgb(colorScale(String(d.source.index)) as string).darker().toString());
  }, [matrix, labels, width, height, colors]);

  return <svg ref={svgRef} width={width} height={height} />;
};

// ============================================
// 6. CIRCULAR BAR PLOT
// ============================================
// File: CircularBarPlot.tsx
interface CircularBarPlotProps {
  data: { label: string; value: number }[];
  width?: number;
  height?: number;
  innerRadius?: number;
  color?: string;
}

export const CircularBarPlot: React.FC<CircularBarPlotProps> = ({
  data,
  width = 600,
  height = 600,
  innerRadius = 100,
  color = '#f56565'
}) => {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || !data.length) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const outerRadius = Math.min(width, height) / 2 - 40;

    const g = svg
      .append('g')
      .attr('transform', `translate(${width / 2},${height / 2})`);

    // Scales
    const xScale = d3
      .scaleBand()
      .domain(data.map(d => d.label))
      .range([0, 2 * Math.PI])
      .align(0);

    const yScale = d3
      .scaleRadial()
      .domain([0, d3.max(data, d => d.value) || 0])
      .range([innerRadius, outerRadius]);

    // Draw bars
    g.selectAll('path')
      .data(data)
      .enter()
      .append('path')
      .attr('fill', color)
      .attr(
        'd',
        d3
          .arc<any>()
          .innerRadius(innerRadius)
          .outerRadius(d => yScale(d.value))
          .startAngle(d => xScale(d.label) || 0)
          .endAngle(d => (xScale(d.label) || 0) + xScale.bandwidth())
          .padAngle(0.01)
          .padRadius(innerRadius)
      );

    // Add labels
    g.selectAll('text')
      .data(data)
      .enter()
      .append('text')
      .attr('text-anchor', d => {
        const angle = ((xScale(d.label) || 0) + xScale.bandwidth() / 2) * (180 / Math.PI);
        return angle > 90 && angle < 270 ? 'end' : 'start';
      })
      .attr('transform', d => {
        const angle = ((xScale(d.label) || 0) + xScale.bandwidth() / 2) * (180 / Math.PI) - 90;
        return `rotate(${angle}) translate(${outerRadius + 10},0) ${
          angle > 90 && angle < 270 ? 'rotate(180)' : ''
        }`;
      })
      .text(d => d.label)
      .style('font-size', '10px');
  }, [data, width, height, innerRadius, color]);

  return <svg ref={svgRef} width={width} height={height} />;
};

// ============================================
// 7. DENSITY PLOT (2D)
// ============================================
// File: DensityPlot.tsx
interface DensityPlotProps {
  data: { x: number; y: number }[];
  width?: number;
  height?: number;
  margin?: { top: number; right: number; bottom: number; left: number };
  bandwidth?: number;
}

export const DensityPlot: React.FC<DensityPlotProps> = ({
  data,
  width = 600,
  height = 600,
  margin = { top: 20, right: 20, bottom: 40, left: 50 },
  bandwidth = 20
}) => {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || !data.length) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const g = svg
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Scales
    const xScale = d3
      .scaleLinear()
      .domain(d3.extent(data, d => d.x) as [number, number])
      .range([0, innerWidth]);

    const yScale = d3
      .scaleLinear()
      .domain(d3.extent(data, d => d.y) as [number, number])
      .range([innerHeight, 0]);

    // Compute density contours
    const densityData = d3
      .contourDensity<{ x: number; y: number }>()
      .x(d => xScale(d.x))
      .y(d => yScale(d.y))
      .size([innerWidth, innerHeight])
      .bandwidth(bandwidth)(data);

    // Color scale
    const colorScale = d3
      .scaleSequential(d3.interpolateYlOrRd)
      .domain([0, d3.max(densityData, d => d.value) || 0]);

    // Draw contours
    g.selectAll('path')
      .data(densityData)
      .enter()
      .append('path')
      .attr('d', d3.geoPath())
      .attr('fill', d => colorScale(d.value))
      .attr('stroke', 'none')
      .attr('opacity', 0.7);

    // Axes
    g.append('g')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(d3.axisBottom(xScale));

    g.append('g').call(d3.axisLeft(yScale));
  }, [data, width, height, margin, bandwidth]);

  return <svg ref={svgRef} width={width} height={height} />;
};

// ============================================
// 8. HEATMAP
// ============================================
// File: Heatmap.tsx
interface HeatmapProps {
  data: { x: string; y: string; value: number }[];
  width?: number;
  height?: number;
  margin?: { top: number; right: number; bottom: number; left: number };
  colorScheme?: readonly string[];
}

export const Heatmap: React.FC<HeatmapProps> = ({
  data,
  width = 600,
  height = 600,
  margin = { top: 40, right: 20, bottom: 60, left: 60 },
  colorScheme = d3.schemeRdYlBu[9]
}) => {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || !data.length) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const g = svg
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Get unique x and y values
    const xValues = Array.from(new Set(data.map(d => d.x)));
    const yValues = Array.from(new Set(data.map(d => d.y)));

    // Scales
    const xScale = d3.scaleBand().domain(xValues).range([0, innerWidth]).padding(0.05);

    const yScale = d3.scaleBand().domain(yValues).range([0, innerHeight]).padding(0.05);

    const colorScale = d3
      .scaleSequential()
      .interpolator(d3.interpolateRdYlBu)
      .domain([d3.max(data, d => d.value) || 0, d3.min(data, d => d.value) || 0]);

    // Draw cells
    g.selectAll('rect')
      .data(data)
      .enter()
      .append('rect')
      .attr('x', d => xScale(d.x) || 0)
      .attr('y', d => yScale(d.y) || 0)
      .attr('width', xScale.bandwidth())
      .attr('height', yScale.bandwidth())
      .attr('fill', d => colorScale(d.value))
      .attr('stroke', 'white')
      .attr('stroke-width', 1);

    // Axes
    g.append('g')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(d3.axisBottom(xScale))
      .selectAll('text')
      .attr('transform', 'rotate(-45)')
      .style('text-anchor', 'end');

    g.append('g').call(d3.axisLeft(yScale));
  }, [data, width, height, margin, colorScheme]);

  return <svg ref={svgRef} width={width} height={height} />;
};

// ============================================
// EXPORT ALL COMPONENTS
// ============================================
export default {
  AreaChart,
  BarChart,
  StreamChart,
  NetworkDiagram,
  ChordDiagram,
  CircularBarPlot,
  DensityPlot,
  Heatmap
};