import { useState, useEffect, useRef } from 'react';
import * as d3 from 'd3';

// Composant principal
export default function USChoroplethMap() {
  const [statesData, setStatesData] = useState(null);
  const [countiesData, setCountiesData] = useState(null);
  const [schoolsData, setSchoolsData] = useState(null);
  const [selectedState, setSelectedState] = useState(null);
  const [loading, setLoading] = useState(true);
  const [hoveredFeature, setHoveredFeature] = useState(null);
  const svgRef = useRef(null);

  // Simuler le chargement des données
  useEffect(() => {
    const loadData = async () => {
      try {
        // Charger directement les données GeoJSON des USA
        const statesResponse = await fetch('https://cdn.jsdelivr.net/npm/us-atlas@3/states-albers-10m.json');
        const statesJson = await statesResponse.json();
        
        // Créer des structures GeoJSON simplifiées pour les états et comtés
        const statesFeatures = statesJson.features;
        
        // Générer des données de comtés simplifiées
        const countiesFeatures = [];
        
        // Générer des données aléatoires pour les établissements scolaires par état
        const stateSchoolsData = statesFeatures.map(state => {
          const stateId = state.id;
          const stateName = getStateName(stateId);
          const schoolCount = Math.floor(Math.random() * 1000) + 100;
          
          return {
            id: stateId,
            name: stateName,
            schoolCount: schoolCount,
            density: schoolCount / (Math.random() * 100 + 50) // Densité simulée
          };
        });
        
        // Générer des données aléatoires pour les écoles par comté
        const countySchoolsData = countiesFeatures.map(county => {
          const countyId = county.id;
          const stateFips = String(countyId).substring(0, 2);
          const schoolCount = Math.floor(Math.random() * 100) + 5;
          
          return {
            id: countyId,
            stateId: stateFips,
            schoolCount: schoolCount,
            schools: Array.from({ length: Math.floor(Math.random() * 10) + 1 }, () => ({
              name: `École ${Math.floor(Math.random() * 1000)}`,
              students: Math.floor(Math.random() * 1500) + 100,
              area: Math.floor(Math.random() * 10000) + 1000,
              lat: county.geometry ? getCentroid(county.geometry)[1] + (Math.random() * 0.5 - 0.25) : 0,
              lng: county.geometry ? getCentroid(county.geometry)[0] + (Math.random() * 0.5 - 0.25) : 0
            }))
          };
        });
        
        setUsData(usTopoJson);
        setStatesData(statesFeatures);
        setCountiesData(countiesFeatures);
        setSchoolsData({ states: stateSchoolsData, counties: countySchoolsData });
        setLoading(false);
      } catch (error) {
        console.error("Erreur lors du chargement des données:", error);
        setLoading(false);
      }
    };
    
    loadData();
  }, []);

  // Rendu de la carte
  useEffect(() => {
    if (loading || !svgRef.current || !statesData || !schoolsData) return;

    const width = 960;
    const height = 600;
    const svg = d3.select(svgRef.current);

    svg.selectAll("*").remove();

    const projection = d3.geoAlbersUsa()
      .fitSize([width, height], selectedState ? 
        statesData.find(state => state.id === selectedState) : 
        { type: "FeatureCollection", features: statesData });

    const path = d3.geoPath().projection(projection);
    
    // Échelle de couleur pour la densité
    const colorScale = d3.scaleSequential(d3.interpolateBlues)
      .domain([0, selectedState ? 
        d3.max(schoolsData.counties.filter(c => c.stateId === selectedState), d => d.schoolCount) : 
        d3.max(schoolsData.states, d => d.density)]);

    // Groupe principal
    const g = svg.append("g");
    
    // Tooltip
    const tooltip = d3.select("body").append("div")
      .attr("class", "absolute hidden bg-white p-2 rounded shadow-lg border border-gray-300 z-10 pointer-events-none")
      .style("opacity", 0);

    if (selectedState) {
      // Afficher les comtés de l'état sélectionné
      const stateCounties = countiesData.filter(county => 
        String(county.id).substring(0, 2) === selectedState);
      
      // Dessiner les contours de l'état
      g.append("path")
        .datum(statesData.find(state => state.id === selectedState))
        .attr("fill", "none")
        .attr("stroke", "#000")
        .attr("stroke-width", 2)
        .attr("d", path);
      
      // Dessiner les comtés
      g.selectAll(".county")
        .data(stateCounties)
        .enter()
        .append("path")
        .attr("class", "county")
        .attr("fill", d => {
          const countyData = schoolsData.counties.find(c => c.id === d.id);
          return countyData ? colorScale(countyData.schoolCount) : "#ccc";
        })
        .attr("d", path)
        .attr("stroke", "#fff")
        .attr("stroke-width", 0.5)
        .on("mouseover", (event, d) => {
          const countyData = schoolsData.counties.find(c => c.id === d.id);
          if (countyData) {
            setHoveredFeature({
              name: d.properties?.name || "Comté",
              schoolCount: countyData.schoolCount,
              type: "county"
            });
            
            tooltip.transition()
              .duration(200)
              .style("opacity", 0.9);
            tooltip.html(`
              <div class="font-bold">${d.properties?.name || "Comté"}</div>
              <div>Établissements: ${countyData.schoolCount}</div>
            `)
              .style("left", (event.pageX + 10) + "px")
              .style("top", (event.pageY - 28) + "px")
              .classed("hidden", false);
          }
        })
        .on("mouseout", () => {
          setHoveredFeature(null);
          tooltip.transition()
            .duration(500)
            .style("opacity", 0)
            .on("end", () => tooltip.classed("hidden", true));
        });
      
      // Afficher les établissements scolaires
      const stateSchools = schoolsData.counties
        .filter(c => c.stateId === selectedState)
        .flatMap(county => county.schools.map(school => ({
          ...school,
          countyId: county.id
        })));
        
      g.selectAll(".school")
        .data(stateSchools)
        .enter()
        .append("circle")
        .attr("class", "school")
        .attr("cx", d => projection([d.lng, d.lat])?.[0])
        .attr("cy", d => projection([d.lng, d.lat])?.[1])
        .attr("r", 3)
        .attr("fill", "red")
        .attr("stroke", "#fff")
        .attr("stroke-width", 0.5)
        .on("mouseover", (event, d) => {
          tooltip.transition()
            .duration(200)
            .style("opacity", 0.9);
          tooltip.html(`
            <div class="font-bold">${d.name}</div>
            <div>Élèves: ${d.students}</div>
            <div>Superficie: ${d.area} m²</div>
          `)
            .style("left", (event.pageX + 10) + "px")
            .style("top", (event.pageY - 28) + "px")
            .classed("hidden", false);
        })
        .on("mouseout", () => {
          tooltip.transition()
            .duration(500)
            .style("opacity", 0)
            .on("end", () => tooltip.classed("hidden", true));
        });
        
      // Bouton retour
      svg.append("g")
        .attr("class", "button")
        .attr("transform", "translate(20, 30)")
        .append("rect")
        .attr("width", 100)
        .attr("height", 30)
        .attr("rx", 5)
        .attr("fill", "#f0f0f0")
        .attr("stroke", "#000")
        .attr("cursor", "pointer")
        .on("click", () => {
          setSelectedState(null);
        });
      
      svg.select(".button")
        .append("text")
        .attr("x", 50)
        .attr("y", 20)
        .attr("text-anchor", "middle")
        .attr("fill", "#000")
        .attr("pointer-events", "none")
        .text("Retour");
        
    } else {
      // Afficher tous les états
      g.selectAll(".state")
        .data(statesData)
        .enter()
        .append("path")
        .attr("class", "state")
        .attr("fill", d => {
          const stateData = schoolsData.states.find(s => s.id === d.id);
          return stateData ? colorScale(stateData.density) : "#ccc";
        })
        .attr("d", path)
        .attr("stroke", "#fff")
        .attr("stroke-width", 0.5)
        .attr("cursor", "pointer")
        .on("click", (event, d) => {
          setSelectedState(d.id);
        })
        .on("mouseover", (event, d) => {
          const stateData = schoolsData.states.find(s => s.id === d.id);
          if (stateData) {
            setHoveredFeature({
              name: stateData.name,
              schoolCount: stateData.schoolCount,
              density: stateData.density.toFixed(2),
              type: "state"
            });
            
            tooltip.transition()
              .duration(200)
              .style("opacity", 0.9);
            tooltip.html(`
              <div class="font-bold">${stateData.name}</div>
              <div>Établissements: ${stateData.schoolCount}</div>
              <div>Densité: ${stateData.density.toFixed(2)}</div>
              <div class="text-xs italic">Cliquez pour zoomer</div>
            `)
              .style("left", (event.pageX + 10) + "px")
              .style("top", (event.pageY - 28) + "px")
              .classed("hidden", false);
          }
        })
        .on("mouseout", () => {
          setHoveredFeature(null);
          tooltip.transition()
            .duration(500)
            .style("opacity", 0)
            .on("end", () => tooltip.classed("hidden", true));
        });
    }
    
    // Légende
    const legendWidth = 200;
    const legendHeight = 20;
    const legend = svg.append("g")
      .attr("transform", `translate(${width - legendWidth - 20}, ${height - 40})`);
    
    const legendScale = d3.scaleLinear()
      .domain(colorScale.domain())
      .range([0, legendWidth]);
    
    const legendAxis = d3.axisBottom(legendScale)
      .ticks(5)
      .tickSize(6);
    
    // Gradient pour la légende
    const defs = svg.append("defs");
    const linearGradient = defs.append("linearGradient")
      .attr("id", "color-gradient")
      .attr("x1", "0%")
      .attr("y1", "0%")
      .attr("x2", "100%")
      .attr("y2", "0%");
    
    // Générer les stops pour le gradient
    const stops = d3.range(0, 1.1, 0.1);
    stops.forEach(stop => {
      linearGradient.append("stop")
        .attr("offset", `${stop * 100}%`)
        .attr("stop-color", colorScale(stop * colorScale.domain()[1]));
    });
    
    // Rectangle avec le gradient
    legend.append("rect")
      .attr("width", legendWidth)
      .attr("height", legendHeight)
      .style("fill", "url(#color-gradient)");
    
    // Axe de la légende
    legend.append("g")
      .attr("transform", `translate(0, ${legendHeight})`)
      .call(legendAxis);
    
    // Titre de la légende
    legend.append("text")
      .attr("y", -5)
      .attr("font-size", "10px")
      .text(selectedState ? "Nombre d'établissements par comté" : "Densité d'établissements par état");
    
    // Nettoyer
    return () => {
      tooltip.remove();
    };
  }, [loading, statesData, countiesData, schoolsData, selectedState]);

  // Fonctions utilitaires
  function getStateName(stateId) {
    const stateNames = {
      "01": "Alabama", "02": "Alaska", "04": "Arizona", "05": "Arkansas", 
      "06": "California", "08": "Colorado", "09": "Connecticut", "10": "Delaware",
      "11": "District of Columbia", "12": "Florida", "13": "Georgia", "15": "Hawaii",
      "16": "Idaho", "17": "Illinois", "18": "Indiana", "19": "Iowa",
      "20": "Kansas", "21": "Kentucky", "22": "Louisiana", "23": "Maine",
      "24": "Maryland", "25": "Massachusetts", "26": "Michigan", "27": "Minnesota",
      "28": "Mississippi", "29": "Missouri", "30": "Montana", "31": "Nebraska",
      "32": "Nevada", "33": "New Hampshire", "34": "New Jersey", "35": "New Mexico",
      "36": "New York", "37": "North Carolina", "38": "North Dakota", "39": "Ohio",
      "40": "Oklahoma", "41": "Oregon", "42": "Pennsylvania", "44": "Rhode Island",
      "45": "South Carolina", "46": "South Dakota", "47": "Tennessee", "48": "Texas",
      "49": "Utah", "50": "Vermont", "51": "Virginia", "53": "Washington",
      "54": "West Virginia", "55": "Wisconsin", "56": "Wyoming"
    };
    return stateNames[stateId] || `État ${stateId}`;
  }
  
  function getCentroid(geometry) {
    if (!geometry) return [0, 0];
    try {
      // Calcul simple du centroïde pour les polygones
      if (geometry.type === "Polygon") {
        const coords = geometry.coordinates[0];
        const x = coords.reduce((sum, coord) => sum + coord[0], 0) / coords.length;
        const y = coords.reduce((sum, coord) => sum + coord[1], 0) / coords.length;
        return [x, y];
      } 
      // Pour les multipolygones, on prend le centroïde du premier polygone
      else if (geometry.type === "MultiPolygon") {
        const coords = geometry.coordinates[0][0];
        const x = coords.reduce((sum, coord) => sum + coord[0], 0) / coords.length;
        const y = coords.reduce((sum, coord) => sum + coord[1], 0) / coords.length;
        return [x, y];
      }
      return [0, 0];
    } catch (e) {
      return [0, 0];
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-xl font-bold">Chargement de la carte...</div>
      </div>
    );
  }

  return (
    <div className="flex flex-col items-center">
      <h1 className="text-2xl font-bold mb-4">
        {selectedState 
          ? `Établissements scolaires en ${schoolsData?.states.find(s => s.id === selectedState)?.name || "État"}`
          : "Densité d'établissements scolaires aux États-Unis"}
      </h1>
      
      {hoveredFeature && (
        <div className="bg-gray-100 p-2 rounded mb-4">
          {hoveredFeature.type === "state" ? (
            <div>
              <span className="font-bold">{hoveredFeature.name}</span>: 
              {" "}Établissements: {hoveredFeature.schoolCount}, 
              Densité: {hoveredFeature.density}
            </div>
          ) : (
            <div>
              <span className="font-bold">{hoveredFeature.name}</span>:
              {" "}Établissements: {hoveredFeature.schoolCount}
            </div>
          )}
        </div>
      )}
      
      <svg 
        ref={svgRef} 
        width="960" 
        height="600" 
        className="border border-gray-300 rounded"
      />
      
      <div className="mt-4 text-sm text-gray-600">
        {selectedState 
          ? "Cliquez sur les établissements pour voir les détails ou sur le bouton 'Retour' pour revenir à la vue nationale."
          : "Cliquez sur un état pour zoomer et voir la répartition des établissements scolaires."}
      </div>
    </div>
  );
}