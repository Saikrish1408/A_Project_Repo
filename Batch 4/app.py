import openrouteservice
import folium
from streamlit_folium import st_folium
import streamlit as st
from datetime import datetime, timedelta
import numpy as np
import random
import requests
import time

st.set_page_config(
    page_title="Route Finder with Whale Optimization",
    page_icon="🗺️",
    layout="wide"
)

if 'routes_data' not in st.session_state:
    st.session_state.routes_data = None
if 'source_coords' not in st.session_state:
    st.session_state.source_coords = None
if 'destination_coords' not in st.session_state:
    st.session_state.destination_coords = None
if 'source_full_name' not in st.session_state:
    st.session_state.source_full_name = None
if 'destination_full_name' not in st.session_state:
    st.session_state.destination_full_name = None
if 'optimized_index' not in st.session_state:
    st.session_state.optimized_index = None
if 'best_route' not in st.session_state:
    st.session_state.best_route = None
if 'route_data_for_table' not in st.session_state:
    st.session_state.route_data_for_table = None
if 'show_results' not in st.session_state:
    st.session_state.show_results = False
if 'processing' not in st.session_state:
    st.session_state.processing = False

client = openrouteservice.Client(key="5b3ce3597851110001cf6248f7450cb5301c4b21ba636260adaf9513")

def geocode_location(location_name):
    try:
        geocode_api = "https://api.openrouteservice.org/geocode/search"
        params = {
            "api_key": "5b3ce3597851110001cf6248f7450cb5301c4b21ba636260adaf9513",
            "text": location_name,
            "size": 5,
            "boundary.country": "IND"
        }
        response = requests.get(geocode_api, params=params)
        data = response.json()
        
        if 'features' in data and len(data['features']) > 0:
            best_match = None
            best_score = -1
            
            for feature in data['features']:
                props = feature['properties']
                name = props.get('name', '')
                region = props.get('region', '')
                country = props.get('country', '')
                confidence = props.get('confidence', 0)
                
                if (location_name.lower() in ['puducherry', 'pondicherry'] and 
                    ('puducherry' in name.lower() or 'pondicherry' in name.lower())):
                    confidence += 0.3
                
                if location_name.lower() == 'auroville' and 'auroville' in name.lower():
                    confidence += 0.3
                
                if 'puducherry' in region.lower() or 'tamil nadu' in region.lower():
                    confidence += 0.2
                
                if confidence > best_score:
                    best_score = confidence
                    best_match = feature
            
            if best_match:
                coordinates = best_match['geometry']['coordinates']
                location_name = best_match['properties'].get('label', location_name)
                return coordinates, location_name
            else:
                coordinates = data['features'][0]['geometry']['coordinates']
                location_name = data['features'][0]['properties'].get('label', location_name)
                return coordinates, location_name
        else:
            st.error(f"Location not found: {location_name}")
            return None, None
    except Exception as e:
        st.error(f"Error geocoding location: {e}")
        return None, None

def display_map(routes_data, source, destination, source_name, destination_name, optimized_index=None):
    midpoint = [(source[1] + destination[1]) / 2, (source[0] + destination[0]) / 2]
    m = folium.Map(location=midpoint, zoom_start=13)

    folium.Marker(
        location=[source[1], source[0]], 
        popup=f"Source: {source_name}", 
        icon=folium.Icon(color="blue")
    ).add_to(m)
    
    folium.Marker(
        location=[destination[1], destination[0]], 
        popup=f"Destination: {destination_name}", 
        icon=folium.Icon(color="green")
    ).add_to(m)

    route_colors = ["blue", "green", "purple", "orange"]

    for i, route_data in enumerate(routes_data):
        color = "red" if i == optimized_index else route_colors[i % len(route_colors)]
        folium.PolyLine(
            locations=[[coord[1], coord[0]] for coord in route_data["coordinates"]],
            color=color,
            weight=5,
            popup=f"Route {i+1}: {route_data['distance'] / 1000:.2f} km, {route_data['duration'] / 60:.2f} mins"
        ).add_to(m)

    st_folium(m, width=None, height=500)

def fetch_route(client, coordinates, avoid_highways=False, avoid_tolls=False):
    try:
        options = {}
        if avoid_highways or avoid_tolls:
            options['avoid_features'] = []
            if avoid_highways:
                options['avoid_features'].append('highways')
            if avoid_tolls:
                options['avoid_features'].append('tollways')
                
        route = client.directions(
            coordinates=coordinates,
            profile="driving-car",
            options=options
        )
        return route
    except Exception as e:
        st.error(f"Error fetching route: {e}")
        return None

def process_route(route):
    if route:
        geometry = route['routes'][0]['geometry']
        decoded_route = openrouteservice.convert.decode_polyline(geometry)
        summary = route['routes'][0]['summary']
        return {
            "coordinates": decoded_route['coordinates'],
            "distance": summary['distance'],
            "duration": summary['duration'],
        }
    return None

def generate_routes_with_waypoints(client, source, destination, avoid_highways=False, avoid_tolls=False):
    direct_route = fetch_route(client, [source, destination], avoid_highways, avoid_tolls)
    routes_data = []
    
    if direct_route:
        route_data = process_route(direct_route)
        if route_data:
            routes_data.append(route_data)
    
    lat_diff = destination[1] - source[1]
    lon_diff = destination[0] - source[0]
    distance_factor = max(0.001, min(0.005, (abs(lat_diff) + abs(lon_diff)) / 10))
    
    variations = [
        [[source[0] + distance_factor, source[1]], destination],
        [[source[0], source[1] + distance_factor], destination],
        [[source[0] + distance_factor/2, source[1] + distance_factor/2], destination],
    ]

    for var in variations:
        coords = [source, *var]
        route = fetch_route(client, coords, avoid_highways, avoid_tolls)
        if route:
            route_data = process_route(route)
            if route_data:
                routes_data.append(route_data)
        time.sleep(1)
        
    return routes_data

def whale_optimization(routes_data, max_iter=30, search_agents=4):
    def objective_function(weights):
        return sum(
            weights[i] * (0.7 * routes_data[i]['distance'] + 0.3 * routes_data[i]['duration'])
            for i in range(len(routes_data))
        )

    distances = [route['distance'] for route in routes_data]
    min_distance = min(distances)
    max_distance = max(distances)
    
    if max_distance > min_distance * 1.5:
        return distances.index(min_distance)
    
    num_routes = len(routes_data)
    positions = np.random.uniform(0, 1, (search_agents, num_routes))
    positions /= positions.sum(axis=1, keepdims=True)

    best_position = positions[0]
    best_score = objective_function(best_position)

    for iteration in range(max_iter):
        for i in range(search_agents):
            a = 2 - iteration * (2 / max_iter)
            r = random.uniform(0, 1)
            A = 2 * a * r - a
            C = 2 * r

            if abs(A) < 1:
                D = abs(C * best_position - positions[i])
                positions[i] = best_position - A * D
            else:
                random_agent = positions[random.randint(0, search_agents - 1)]
                D = abs(C * random_agent - positions[i])
                positions[i] = random_agent - A * D

            score = objective_function(positions[i])
            if score < best_score:
                best_score = score
                best_position = positions[i]

        positions = np.clip(positions, 0, 1)
        positions /= positions.sum(axis=1, keepdims=True)

    best_index = np.argmax(best_position)
    return best_index

def calculate_eta(duration):
    arrival_time = datetime.now() + timedelta(seconds=duration)
    return arrival_time.strftime("%I:%M %p")

def process_routing():
    st.session_state.processing = True
    
    st.session_state.source_coords, st.session_state.source_full_name = geocode_location(st.session_state.source)
    st.session_state.destination_coords, st.session_state.destination_full_name = geocode_location(st.session_state.destination)
    
    if st.session_state.source_coords and st.session_state.destination_coords:
        st.session_state.routes_data = generate_routes_with_waypoints(
            client, 
            st.session_state.source_coords, 
            st.session_state.destination_coords,
            st.session_state.avoid_highways,
            st.session_state.avoid_tolls
        )
        
        if st.session_state.routes_data:
            st.session_state.route_data_for_table = []
            for i, route in enumerate(st.session_state.routes_data):
                st.session_state.route_data_for_table.append({
                    "Route": f"Route {i+1}",
                    "Distance (km)": f"{route['distance'] / 1000:.2f}",
                    "Duration (mins)": f"{route['duration'] / 60:.2f}"
                })
            
            st.session_state.optimized_index = whale_optimization(st.session_state.routes_data)
            st.session_state.best_route = st.session_state.routes_data[st.session_state.optimized_index]
            st.session_state.show_results = True
        else:
            st.error("Could not fetch any routes. Please verify locations and API settings.")
            st.session_state.show_results = False
    else:
        st.error("Could not geocode one or both locations. Please check the location names and try again.")
        st.session_state.show_results = False
    
    st.session_state.processing = False

def reset_search():
    for key in [
        'routes_data', 'source_coords', 'destination_coords', 
        'source_full_name', 'destination_full_name', 
        'optimized_index', 'best_route', 'route_data_for_table',
        'show_results'
    ]:
        if key in st.session_state:
            st.session_state[key] = None
    
    st.session_state.show_results = False

if not st.session_state.show_results:
    st.title("Route Finder with Whale Optimization")
    
    with st.form("route_form"):
        st.session_state.source = st.text_input("Source Location (e.g., Puducherry)", "Puducherry")
        st.session_state.destination = st.text_input("Destination Location (e.g., Auroville)", "Auroville")
        
        st.subheader("Route Preferences")
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.avoid_highways = st.checkbox("Avoid Highways")
        with col2:
            st.session_state.avoid_tolls = st.checkbox("Avoid Toll Roads")
        
        submit_button = st.form_submit_button("Find Optimized Route")
        
        if submit_button:
            process_routing()
            st.rerun()
else:
    st.title("Optimized Route Results")
    
    if st.button("← Start a New Search"):
        reset_search()
        st.rerun()
    
    st.header("Route Information")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**From:** {st.session_state.source_full_name}")
        st.markdown(f"**Location:** Longitude: {st.session_state.source_coords[0]}, Latitude: {st.session_state.source_coords[1]}")
        
    with col2:
        st.markdown(f"**To:** {st.session_state.destination_full_name}")
        st.markdown(f"**Location:** Longitude: {st.session_state.destination_coords[0]}, Latitude: {st.session_state.destination_coords[1]}")
    
    st.subheader("Available Routes")
    st.dataframe(st.session_state.route_data_for_table, use_container_width=True)
    
    best_route = st.session_state.best_route
    optimized_index = st.session_state.optimized_index
    
    st.subheader("Optimized Route")
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    
    with metrics_col1:
        st.metric("Distance", f"{best_route['distance'] / 1000:.2f} km")
    
    with metrics_col2:
        st.metric("Duration", f"{best_route['duration'] / 60:.2f} minutes")
    
    with metrics_col3:
        st.metric("Estimated Arrival", calculate_eta(best_route['duration']))
    
    st.subheader("Route Map")
    st.info("The **red** route is the optimized path recommended by the Whale Optimization Algorithm.")
    
    display_map(
        st.session_state.routes_data, 
        st.session_state.source_coords, 
        st.session_state.destination_coords, 
        st.session_state.source_full_name, 
        st.session_state.destination_full_name, 
        optimized_index=st.session_state.optimized_index
    )

st.markdown("---")
st.markdown("Route Finder with Whale Optimization Algorithm | © 2023")