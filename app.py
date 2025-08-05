from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import time
from math import radians, cos, sin, sqrt, atan2
from geopy.distance import geodesic
from openai import OpenAI
import datetime
import logging
import os
import random
import string
from datetime import datetime
import json
from dotenv import load_dotenv

app = Flask(__name__) 


CORS(app)



# API Keys
GOOGLE_API_KEY = 'AIzaSyDsfHwAB8GKbmiZu8n40d3M6n0ZtsCzwcg'
OCM_API_KEY = "d1f80a51-a153-4483-916e-e5b56e806ed1"
OSRM_URL = "http://router.project-osrm.org/route/v1/driving"
OCM_API_KEY_1 = "6026746e-34b3-43bf-af62-2b4f91ec9a77"
client = OpenAI(api_key="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")
# EV Database
EV_DATABASE = {
    "Tata Nexon EV": {"make": "Tata", "model": "Nexon EV", "electric_range": 312},
    "Hyundai Kona Electric": {"make": "Hyundai", "model": "Kona Electric", "electric_range": 452},
    "MG ZS EV": {"make": "MG", "model": "ZS EV", "electric_range": 419},
    "Mahindra XUV400": {"make": "Mahindra", "model": "XUV400", "electric_range": 375},
    "BYD Atto 3": {"make": "BYD", "model": "Atto 3", "electric_range": 521},
    "Kia EV6": {"make": "Kia", "model": "EV6", "electric_range": 528}
}
# Add this to your Flask backend file
EV_SPECS = {
    "Tata Nexon EV": {"make": "Tata", "model": "Nexon EV", "battery_capacity_kwh": 30.2},
    "Tata Nexon EV Max": {"make": "Tata", "model": "Nexon EV Max", "battery_capacity_kwh": 40.5},
    "Hyundai Kona Electric": {"make": "Hyundai", "model": "Kona Electric", "battery_capacity_kwh": 39.2},
    "MG ZS EV": {"make": "MG", "model": "ZS EV", "battery_capacity_kwh": 50.3},
    "Mahindra XUV400": {"make": "Mahindra", "model": "XUV400", "battery_capacity_kwh": 39.4},
    "BYD Atto 3": {"make": "BYD", "model": "Atto 3", "battery_capacity_kwh": 60.5},
    "Kia EV6": {"make": "Kia", "model": "EV6", "battery_capacity_kwh": 77.4}
}
# --- Helper Functions ---
def get_ev_max_range(make, model):
    for entry in EV_DATABASE.values():
        if entry["make"].lower() == make.lower() and entry["model"].lower() == model.lower():
            return entry["electric_range"]
    return None

def haversine_km(p1, p2):
    R = 6371  # Earth radius in km
    lat1, lon1 = radians(p1[0]), radians(p1[1])
    lat2, lon2 = radians(p2[0]), radians(p2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

# --- Route Calculation ---
def get_route_coordinates(src, dst):
    """Get route coordinates from OSRM"""
    url = f"{OSRM_URL}/{src[1]},{src[0]};{dst[1]},{dst[0]}?overview=full&geometries=geojson"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data["routes"]:
            return data["routes"][0]["geometry"]["coordinates"]
    return []

def get_route_polyline(origin, destination):
    """Get route from Google Directions API"""
    url = "https://routes.googleapis.com/directions/v2:computeRoutes"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GOOGLE_API_KEY,
        "X-Goog-FieldMask": "routes.polyline.encodedPolyline,routes.distanceMeters"
    }
    data = {
        "origin": {"location": {"latLng": {"latitude": float(origin.split(',')[0]), "longitude": float(origin.split(',')[1])}}},
        "destination": {"location": {"latLng": {"latitude": float(destination.split(',')[0]), "longitude": float(destination.split(',')[1])}}},
        "travelMode": "DRIVE"
    }
    response = requests.post(url, headers=headers, json=data)
    response_data = response.json()

    if "routes" not in response_data or not response_data["routes"]:
        raise Exception("Routes API error: No routes found")

    total_distance_meters = response_data["routes"][0].get("distanceMeters", 0)
    return decode_polyline(response_data["routes"][0]["polyline"]["encodedPolyline"]), total_distance_meters

def decode_polyline(polyline_str):
    """Decode Google's polyline encoding"""
    index, lat, lng, coordinates = 0, 0, 0, []
    while index < len(polyline_str):
        result, shift = 1, 0
        while True:
            b = ord(polyline_str[index]) - 63 - 1
            index += 1
            result += b << shift
            shift += 5
            if b < 0x1f:
                break
        lat += ~(result >> 1) if result & 1 else result >> 1

        result, shift = 1, 0
        while True:
            b = ord(polyline_str[index]) - 63 - 1
            index += 1
            result += b << shift
            shift += 5
            if b < 0x1f:
                break
        lng += ~(result >> 1) if result & 1 else result >> 1

        coordinates.append((lat * 1e-5, lng * 1e-5))
    return coordinates

# --- Station Search ---
# Replace the current error handling in get_charging_stations_near_point with:
def get_charging_stations_near_point(lat, lon, radius_km=15):
    """Search OpenChargeMap for stations with better error handling"""
    try:
        url = "https://api.openchargemap.io/v3/poi/"
        params = {
            "output": "json",
            "latitude": lat,
            "longitude": lon,
            "distance": radius_km,
            "distanceunit": "KM",
            "maxresults": 10,
            "compact": "true",
            "verbose": "false",
            "key": OCM_API_KEY
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.RequestException as e:
        logging.error(f"OCM API request failed: {str(e)}")
        return []
    except ValueError as e:
        logging.error(f"OCM API response parsing failed: {str(e)}")
        return []
def search_ev_stations_nearby(lat, lng, radius=3000):
    """Search Google Places for stations"""
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "location": f"{lat},{lng}",
        "radius": radius,
        "keyword": "EV charging station",
        "key": GOOGLE_API_KEY
    }
    response = requests.get(url, params=params).json()
    return response.get("results", [])

def deduplicate_places(places):
    seen = set()
    return [p for p in places if p["place_id"] not in seen and not seen.add(p["place_id"])]

# --- Route Planning Algorithms ---
def generate_enhanced_waypoints(route_coords, range_km, search_window=5):
    """Generate charging waypoints with fallback logic"""
    waypoints = []
    accumulated_distance = 0.0
    last_coord = (route_coords[0][1], route_coords[0][0])
    i = 1

    while i < len(route_coords):
        current_coord = (route_coords[i][1], route_coords[i][0])
        segment_distance = geodesic(last_coord, current_coord).km
        accumulated_distance += segment_distance

        if accumulated_distance >= range_km:
            found = False
            fallback_index = None
            fallback_station = None

            for offset in range(-search_window, search_window + 1):
                j = i + offset
                if 0 <= j < len(route_coords):
                    lat, lon = route_coords[j][1], route_coords[j][0]
                    stations = get_charging_stations_near_point(lat, lon)
                    if stations:
                        waypoints.append(((lat, lon), stations))
                        i = j
                        found = True
                        break
                    elif fallback_index is None:
                        fallback_index = j
                        fallback_station = (lat, lon)

            if not found and fallback_station:
                waypoints.append((fallback_station, []))
                i = fallback_index

            accumulated_distance = 0.0
            last_coord = current_coord
        else:
            last_coord = current_coord
            i += 1

    return waypoints

def find_minimum_stops(polyline, stations, current_range_km, max_range_km):
    """Calculate minimum required stops"""
    if not polyline:
        return [], 0

    station_points = [(s['geometry']['location']['lat'], 
                       s['geometry']['location']['lng'], s) 
                     for s in stations if s.get('geometry', {}).get('location')]

    recommended = []
    last_charge_point = polyline[0]
    stop_count = 0

    for i in range(len(polyline) - 1):
        segment_end = polyline[i+1]
        distance_from_last_charge = haversine_km(last_charge_point, segment_end)

        if distance_from_last_charge > max_range_km:
            eligible_stations = []
            for s_lat, s_lng, station_data in station_points:
                station_pos = (s_lat, s_lng)
                if (haversine_km(last_charge_point, station_pos) <= max_range_km and
                    haversine_km(station_pos, polyline[-1]) < haversine_km(last_charge_point, polyline[-1])):
                    eligible_stations.append((haversine_km(station_pos, polyline[-1]), station_data, station_pos))

            if not eligible_stations:
                return [], -1

            best_station = eligible_stations[0][1]
            best_station_coords = eligible_stations[0][2]
            
            recommended.append(best_station)
            last_charge_point = best_station_coords
            stop_count += 1

    # Final destination check
    if haversine_km(last_charge_point, polyline[-1]) > max_range_km:
        eligible_stations = [
            (haversine_km((s_lat, s_lng), polyline[-1]), station_data, (s_lat, s_lng))
            for s_lat, s_lng, station_data in station_points
            if (haversine_km(last_charge_point, (s_lat, s_lng)) <= max_range_km and
                haversine_km((s_lat, s_lng), polyline[-1]) < max_range_km)
        ]
        
        if eligible_stations:
            recommended.append(eligible_stations[0][1])
            stop_count += 1
        else:
            return [], -1

    return recommended, stop_count

# --- API Endpoints ---
@app.route('/api/estimate/range_and_battery_analysis', methods=['POST'])
def estimate_range_and_battery():
    """
    This new route uses OpenAI to estimate the remaining range in kilometers
    and provide a brief analysis of the battery's state.
    """
    if not client:
        return jsonify({"error": "OpenAI client not initialized. Check your API key."}), 500

    # 1. Validate incoming data
    data = request.get_json()
    if not data or 'evModel' not in data or 'batteryRemaining' not in data:
        return jsonify({"error": "Missing required fields: evModel and batteryRemaining"}), 400

    ev_model = data['evModel']
    battery_percentage = data['batteryRemaining']

    # 2. Engineer a new, detailed prompt for the AI
    prompt_messages = [
        {
            "role": "system",
            "content": """
            You are an expert EV analyst. Your task is to provide a detailed battery and range estimation based on vehicle data.
            You must use your knowledge of the specific EV model to estimate its total real-world range.
            From that, calculate the estimated remaining range in kilometers based on the current battery percentage.
            Also, provide a brief, helpful analysis of the battery's current state (e.g., 'Your battery is healthy for daily commutes,' or 'Charge is critically low, immediate charging is advised.').
            You MUST respond ONLY with a valid JSON object in the following format, with no other text or explanation:
            {"estimatedRangeKm": <number>, "batteryAnalysis": "<string>", "rangeConfidence": "<string: 'High'|'Medium'|'Low'>"}
            """
        },
        {
            "role": "user",
            "content": f"""
            Analyze the following EV's status:
            - EV Model: {ev_model}
            - Current Battery Percentage: {battery_percentage}%

            Provide the estimated remaining range in kilometers, a brief analysis of the battery state, and your confidence level in the estimate.
            Return the result in the required JSON format.
            """
        }
    ]

    # 3. Call the OpenAI API
    try:
        print(f"Sending Range Analysis request to OpenAI for model: {ev_model}...")
        
        completion = client.chat.completions.create(
            model="gpt-4o-mini", # Using a newer, more capable model
            messages=prompt_messages,
            response_format={"type": "json_object"},
            temperature=0.3
        )

        # 4. Parse and validate the AI's response
        ai_response_dict = json.loads(completion.choices[0].message.content)

        if "estimatedRangeKm" not in ai_response_dict or "batteryAnalysis" not in ai_response_dict:
            raise ValueError("AI response did not contain the required keys.")

        print(f"Received successful range analysis from OpenAI: {ai_response_dict}")
        return jsonify(ai_response_dict)

    except Exception as e:
        print(f"An error occurred during range analysis API call: {e}")
        return jsonify({"error": "Failed to get range analysis from AI service."}), 500

@app.route('/api/estimate_battery', methods=['POST'])
def estimate_battery():
    try:
        data = request.get_json()
        
        # Required fields validation
        required_fields = ['mode', 'ev_make', 'ev_model', 'current_battery']
        if not all(field in data for field in required_fields):
            return jsonify({
                "status": "error",
                "error": "Missing required fields",
                "required_fields": required_fields
            }), 400

        mode = data['mode']
        ev_make = data['ev_make'].strip()
        ev_model = data['ev_model'].strip()
        current_battery = float(data['current_battery'])

        # Get EV range from database or AI
        max_range = get_ev_range(ev_make, ev_model)
        if not max_range:
            return jsonify({
                "status": "error",
                "error": f"Could not determine range for {ev_make} {ev_model}"
            }), 404

        # Initialize result
        result = {
            "estimated_battery": current_battery,
            "estimated_range": (current_battery / 100) * max_range,
            "max_range": max_range,
            "ev_model": f"{ev_make} {ev_model}"
        }

        # Mode-specific calculations
        if mode == "idle":
            hours_idle = float(data.get('hours_idle', 1))
            temp_factor = 1.5 if data.get('temperature', 25) < 10 else 1.0
            battery_drain = min(0.1 * hours_idle * temp_factor, 5)  # Max 5% drain
            result["estimated_battery"] = max(0, current_battery - battery_drain)
            result["battery_drain"] = battery_drain

        elif mode == "driving":
            driving_style = data.get('driving_style', 'mixed')
            duration_min = float(data.get('duration_min', 0))
            
            efficiency = {
                'city': {'speed': 30, 'factor': 1.2},
                'highway': {'speed': 80, 'factor': 1.1},
                'mixed': {'speed': 50, 'factor': 1.0}
            }.get(driving_style)
            
            distance = (efficiency['speed'] * duration_min) / 60
            battery_used = (distance / max_range) * 100 * efficiency['factor']
            result["estimated_battery"] = max(0, current_battery - battery_used)
            result["distance_covered"] = distance

        elif mode == "gps":
            try:
                start = parse_coordinates(data['start_coords'])
                end = parse_coordinates(data['end_coords'])
                distance = haversine_km(start, end)
                
                elevation_factor = 1 + (float(data.get('elevation_gain', 0)) / 1000 * 0.1)
                battery_used = (distance / max_range) * 100 * elevation_factor
                result["estimated_battery"] = max(0, current_battery - battery_used)
                result["distance_covered"] = distance
            except Exception as e:
                return jsonify({"error": f"Invalid coordinates: {str(e)}"}), 400

        # Generate charging suggestions if battery is low
        if result["estimated_battery"] < 20:
            result["charging_suggestions"] = [
                "Consider charging soon to avoid range anxiety",
                "Look for fast charging stations if traveling far",
                "Precondition battery while plugged in for faster charging"
            ]

        return jsonify({
            "status": "success",
            "data": result
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

def get_ev_range(make, model):
    """Get EV range from database or AI"""
    EV_DATABASE = {
        ("tata", "nexon ev"): 312,
        ("tata", "nexon ev max"): 453,
        ("hyundai", "kona electric"): 452,
        # ... other models
    }
    
    # Try database first
    range_km = EV_DATABASE.get((make.lower(), model.lower()))
    if range_km:
        return range_km
        
    # Fallback to AI (implementation would call OpenAI API)
    return estimate_range_with_ai(make, model)

def parse_coordinates(coords):
    """Parse coordinates from string or list"""
    if isinstance(coords, str):
        parts = [float(x.strip()) for x in coords.split(',')]
        return parts[:2]
    elif isinstance(coords, list) and len(coords) >= 2:
        return [float(coords[0]), float(coords[1])]
    raise ValueError("Invalid coordinate format")




# Add this new route to your Flask backend file
@app.route('/api/estimate/idle', methods=['POST'])
def estimate_idle_with_ai():
    """
    This route uses the OpenAI API to estimate the idle standby time for an EV.
    """
    if not client:
        return jsonify({"error": "OpenAI client not initialized. Check your API key."}), 500

    # 1. Validate the incoming request data
    data = request.get_json()
    if not data or 'evModel' not in data or 'batteryPercentage' not in data or 'evName' not in data:
        return jsonify({"error": "Missing required fields: evName, evModel, and batteryPercentage"}), 400

    ev_name = data['evName']
    ev_model = data['evModel']
    battery_percentage = data['batteryPercentage']

    # 2. Engineer the prompt for the AI
    # This is the most important part. We instruct the AI to be an expert
    # and to return the data in a specific JSON format.
    prompt_messages = [
        {
            "role": "system",
            "content": """
            You are an expert EV technical assistant. Your task is to provide accurate estimations based on vehicle data. 
            You must use your knowledge of electric vehicles to find the total battery capacity (in kWh) and an estimated idle power consumption (in Watts) for the given model.
            Then, calculate how long the vehicle can remain on standby.
            You must respond ONLY with a valid JSON object in the following format, with no other text or explanation:
            {"estimatedDurationHours": <number>, "message": "<string>"}
            """
        },
        {
            "role": "user",
            "content": f"""
            Please provide an idle standby time estimation for the following electric vehicle:
            - EV Name: {ev_name}
            - EV Model: {ev_model}
            - Current Battery Percentage: {battery_percentage}%

            Calculate the result and provide it in the required JSON format.
            """
        }
    ]

    # 3. Call the OpenAI API
    try:
        print(f"Sending request to OpenAI for model: {ev_model}...")
        
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo", # A fast and capable model for this task
            messages=prompt_messages,
            response_format={"type": "json_object"}, # This forces the output to be JSON
            temperature=0.2 # Lower temperature for more deterministic, factual responses
        )

        # 4. Parse the AI's response
        ai_response_str = completion.choices[0].message.content
        # The response content is a JSON string, so we need to parse it into a Python dict
        ai_response_dict = json.loads(ai_response_str)

        # 5. Validate the AI's JSON response to ensure it followed instructions
        if "estimatedDurationHours" not in ai_response_dict or "message" not in ai_response_dict:
            raise ValueError("AI response did not contain the required keys.")

        print(f"Received successful response from OpenAI: {ai_response_dict}")
        return jsonify(ai_response_dict)

    except Exception as e:
        print(f"An error occurred with the OpenAI API call: {e}")
        return jsonify({"error": "Failed to get estimation from AI service."}), 500
@app.route('/personalized_suggestions', methods=['POST'])
def personalized_suggestions():
    try:
        data = request.get_json()
        
        # Required fields
        required_fields = ['battery_percent', 'wallet_balance', 'coupons', 'green_credits']
        if not all(field in data for field in required_fields):
            return jsonify({"error": "Missing required fields"}), 400

        battery_percent = float(data['battery_percent'])
        wallet_balance = float(data['wallet_balance'])
        coupons = data['coupons']  # Array of coupon objects
        green_credits = float(data['green_credits'])
        ev_model = data.get('ev_model', 'your EV')

        # Prepare the prompt for OpenAI
        prompt = f"""Act as an EV charging expert assistant. Provide personalized charging suggestions based on:
        
        User Context:
        - Battery Level: {battery_percent}%
        - Wallet Balance: ₹{wallet_balance}
        - Available Green Credits: {green_credits}
        - Active Coupons: {len([c for c in coupons if not c['used']])}
        - EV Model: {ev_model}

        Provide 3-5 specific, actionable recommendations considering:
        1. Best charging practices for battery health
        2. Optimal use of available coupons
        3. Wallet balance considerations
        4. Green credits redemption opportunities
        5. Current battery level urgency

        Format each suggestion with an emoji and keep it concise (1 sentence max per suggestion).
        """

        # Get AI response
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an EV charging expert assistant providing personalized recommendations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=300
        )

        # Process the response
        suggestions = response.choices[0].message.content.split('\n')
        suggestions = [s.strip() for s in suggestions if s.strip()]

        return jsonify({
            "suggestions": suggestions,
            "battery_status": "critical" if battery_percent < 15 else 
                            "low" if battery_percent < 30 else 
                            "medium" if battery_percent < 80 else 
                            "high",
            "has_active_coupons": len([c for c in coupons if not c['used']]) > 0,
            "can_redeem_credits": green_credits >= 100
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route('/api/forecast', methods=['POST'])
def generate_forecast():
    try:
        data = request.get_json()
        historical_data = data.get('historicalData', [])
        forecast_range = data.get('forecastRange', 'week')
        reference_date = data.get('referenceDate')
        time_slots = data.get('timeSlots', {})
        days_of_week = data.get('daysOfWeek', {})

        # Basic analysis of historical data
        total_sessions = len(historical_data)
        avg_sessions_per_day = total_sessions / max(len(days_of_week), 1)
        
        # Calculate peak time (most frequent time slot)
        peak_time = max(time_slots.items(), key=lambda x: x[1])[0] if time_slots else "7:00 PM"
        
        # Calculate revenue (assuming ₹30 per session on average)
        avg_revenue_per_session = 30
        total_revenue = total_sessions * avg_revenue_per_session
        
        # Generate forecast based on range
        if forecast_range == 'day':
            predicted_sessions = round(avg_sessions_per_day * 1.1)  # 10% growth
            revenue_forecast = predicted_sessions * avg_revenue_per_session
            forecast_chart = [
                {"day": "Morning", "predicted": round(predicted_sessions * 0.3)},
                {"day": "Afternoon", "predicted": round(predicted_sessions * 0.2)},
                {"day": "Evening", "predicted": round(predicted_sessions * 0.5)}
            ]
        else:
            days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            if forecast_range == 'month':
                days = [f"Week {i+1}" for i in range(4)]
            
            # Weight days based on historical patterns
            day_weights = {
                'Mon': days_of_week.get('Monday', 1),
                'Tue': days_of_week.get('Tuesday', 1),
                'Wed': days_of_week.get('Wednesday', 1),
                'Thu': days_of_week.get('Thursday', 1),
                'Fri': days_of_week.get('Friday', 1),
                'Sat': days_of_week.get('Saturday', 1),
                'Sun': days_of_week.get('Sunday', 1)
            }
            
            total_weight = sum(day_weights.values())
            predicted_sessions = round(avg_sessions_per_day * len(days) * 1.1)  # 10% growth
            revenue_forecast = predicted_sessions * avg_revenue_per_session
            
            forecast_chart = [
                {
                    "day": day,
                    "predicted": round(predicted_sessions * (day_weights[day if day in day_weights else day.split()[-1]] / total_weight)),
                    "actual": None
                } for day in days
            ]

        # Generate AI insights
        ai_prompt = f"""Generate EV charging insights based on:
        - Historical sessions: {total_sessions}
        - Peak time: {peak_time}
        - Forecast range: {forecast_range}
        - Reference date: {reference_date}
        
        Provide:
        1. High demand alert (specific day/time)
        2. Optimal maintenance window
        3. Revenue opportunity suggestion
        4. Weather impact prediction (be creative)
        """
        
        ai_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an EV charging expert providing insights and predictions."},
                {"role": "user", "content": ai_prompt}
            ],
            temperature=0.7,
            max_tokens=300
        )
        
        insights_text = ai_response.choices[0].message.content
        insights_list = [i.strip() for i in insights_text.split('\n') if i.strip()]
        
        response = {
            "predictedSessions": predicted_sessions,
            "peakLoadTime": peak_time,
            "revenueForecast": revenue_forecast,
            "forecastChart": forecast_chart,
            "aiInsights": {
                "highDemand": insights_list[0] if len(insights_list) > 0 else "Expected high demand during evening hours.",
                "maintenance": insights_list[1] if len(insights_list) > 1 else "Recommended maintenance during low-usage hours.",
                "revenue": insights_list[2] if len(insights_list) > 2 else "Dynamic pricing could increase revenue by 10-15%.",
                "weather": insights_list[3] if len(insights_list) > 3 else "Weather conditions may impact charging demand."
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route('/api/usage-trends', methods=['POST'])
def get_usage_trends():
    try:
        data = request.get_json()
        bookings = data.get('bookings', [])
        time_range = data.get('timeRange', 'week')

        # Process bookings data
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        hours = list(range(24))
        
        # Initialize data structures
        heatmap = [[0 for _ in hours] for _ in days]
        daily_usage = {day: {'sessions': 0, 'revenue': 0} for day in days}
        
        # Process each booking
        for booking in bookings:
            try:
                booked_at = datetime.fromisoformat(booking['bookedAt'])
                day_index = booked_at.weekday()  # 0=Monday, 6=Sunday
                day_name = days[day_index]
                
                # Parse time (assuming format like "05:00 PM")
                time_str = booking['slotTime']
                time_parts = time_str.split()
                hour_min = time_parts[0].split(':')
                hour = int(hour_min[0])
                is_pm = time_parts[1].upper() == 'PM'
                
                # Convert to 24-hour format
                if is_pm and hour != 12:
                    hour += 12
                elif not is_pm and hour == 12:
                    hour = 0
                
                # Update heatmap
                heatmap[day_index][hour] += 1
                
                # Update daily usage (₹30 per session)
                daily_usage[day_name]['sessions'] += 1
                daily_usage[day_name]['revenue'] += 30
            except Exception as e:
                print(f"Error processing booking: {e}")
                continue

        # Normalize heatmap data (0-1 scale)
        max_sessions = max(max(row) for row in heatmap) or 1
        normalized_heatmap = [[val / max_sessions for val in row] for row in heatmap]

        # Generate forecast using AI
        ai_prompt = f"""Analyze this EV charging station usage data:
        - Days: {days}
        - Hourly data: {heatmap}
        - Daily sessions: {[daily_usage[day]['sessions'] for day in days]}
        - Time range: {time_range}

        Provide:
        1. Busiest day and hour
        2. Three peak periods (morning, evening, night)
        3. One actionable recommendation for station optimization
        """
        
        ai_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an EV charging analytics expert."},
                {"role": "user", "content": ai_prompt}
            ],
            temperature=0.7,
            max_tokens=300
        )
        
        insights = ai_response.choices[0].message.content.split('\n')
        
        # Prepare response
        response = {
            "heatmap": {
                "days": days,
                "hours": hours,
                "data": normalized_heatmap
            },
            "dailyUsage": [{
                "day": day,
                "sessions": daily_usage[day]['sessions'],
                "revenue": daily_usage[day]['revenue']
            } for day in days],
            "forecast": [{
                "day": day,
                "predicted": int(daily_usage[day]['sessions'] * 1.1)  # 10% growth
            } for day in days],
            "peakHours": [
                {
                    "period": "Morning Peak",
                    "time": "7:00 - 9:00 AM",
                    "utilization": int(max(normalized_heatmap[day_idx][7:10]) * 100),
                    "color": "#16FFBD"
                },
                {
                    "period": "Evening Peak",
                    "time": "6:00 - 8:00 PM",
                    "utilization": int(max(normalized_heatmap[day_idx][18:20]) * 100),
                    "color": "#FCEE09"
                },
                {
                    "period": "Night Low",
                    "time": "11:00 PM - 5:00 AM",
                    "utilization": int(max(normalized_heatmap[day_idx][23] + normalized_heatmap[day_idx][0:5]) * 100),
                    "color": "#FF6EC7"
                }
            ],
            "insights": {
                "busiestDay": insights[0].split(':')[-1].strip() if len(insights) > 0 else "Friday",
                "busiestHour": insights[1].split(':')[-1].strip() if len(insights) > 1 else "7:00 PM",
                "recommendation": insights[2].split(':')[-1].strip() if len(insights) > 2 else "Increase capacity during peak hours"
            }
        }

        return jsonify(response)

    except Exception as e:
        print(f"Error in usage trends: {str(e)}")
        return jsonify({"error": str(e)}), 500
@app.route('/process_voice', methods=['POST'])
def process_voice():
    try:
        audio_file = request.files.get('audio')
        if not audio_file:
            return jsonify({"error": "No audio file provided"}), 400

        # Save temporary audio file
        temp_path = os.path.join('/tmp', audio_file.filename)
        audio_file.save(temp_path)

        # Transcribe using OpenAI Whisper
        with open(temp_path, "rb") as audio:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio,
                response_format="text"
            )

        # Clean up
        os.remove(temp_path)

        return jsonify({
            "text": transcript,
            "status": "success"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route('/generate_coupon', methods=['POST'])
def generate_coupon():
    """Generate a new coupon for a user"""
    try:
        data = request.get_json()
        uid = data.get('uid')
        
        if not uid:
            return jsonify({"error": "User ID is required"}), 400
        
        # Generate coupon code
        code = f"ZYNTRA-{''.join(random.choices(string.ascii_uppercase + string.digits, k=8))}"
        discount = random.randint(5, 20)  # 5-20% discount
        valid_until = datetime.datetime.now() + datetime.timedelta(days=30)  # Valid for 30 days
        
        coupon = {
            "code": code,
            "discount": discount,
            "validUntil": valid_until.isoformat(),
            "used": False
        }
        
        # Update user document in Firestore
        user_ref = db.collection('userProfiles').document(uid)
        user_ref.update({
            "coupons": firestore.ArrayUnion([coupon]),
            "greenCredits": firestore.Increment(-100)  # Deduct 100 credits
        })
        
        return jsonify({
            "success": True,
            "coupon": coupon
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route('/find_ev_stations', methods=['POST'])
def api_find_ev_stations():
    """Main endpoint for EV route planning with enhanced error handling"""
    try:
        # ===== 1. Request Validation =====
        print("\n=== New Request Received ===")
        data = request.get_json()
        
        if not data:
            print("Error: No JSON data received")
            return jsonify({"error": "Request body must be JSON"}), 400

        required_fields = ['origin', 'destination', 'current_range_km', 'max_range_km']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            print(f"Error: Missing required fields - {missing_fields}")
            return jsonify({"error": f"Missing required fields: {', '.join(missing_fields)}"}), 400

        print(f"Origin: {data['origin']}")
        print(f"Destination: {data['destination']}")
        print(f"Current Range: {data['current_range_km']}km")
        print(f"Max Range: {data['max_range_km']}km")

        # ===== 2. Coordinate Validation =====
        try:
            origin_lat, origin_lng = map(float, data['origin'].split(','))
            dest_lat, dest_lng = map(float, data['destination'].split(','))
            
            if not (-90 <= origin_lat <= 90) or not (-180 <= origin_lng <= 180):
                raise ValueError("Origin coordinates out of range")
            if not (-90 <= dest_lat <= 90) or not (-180 <= dest_lng <= 180):
                raise ValueError("Destination coordinates out of range")
                
        except ValueError as e:
            print(f"Invalid coordinates format: {str(e)}")
            return jsonify({"error": f"Invalid coordinates format: {str(e)}"}), 400

        # ===== 3. Get Route from Google Directions API =====
        print("\nCalling Google Directions API...")
        try:
            route_polyline, total_distance_meters = get_route_polyline(data['origin'], data['destination'])
            total_distance_km = total_distance_meters / 1000
            print(f"Route obtained - {len(route_polyline)} points, {total_distance_km:.1f}km")
        except Exception as e:
            print(f"Google Directions API failed: {str(e)}")
            return jsonify({
                "error": "Failed to get route from Google Directions API",
                "details": str(e)
            }), 502

        # ===== 4. Find Charging Stations =====
        print("\nSearching for charging stations...")
        all_stations = []
        sampled_points = [route_polyline[i] for i in range(0, len(route_polyline), max(1, len(route_polyline)//10))]
        
        for i, (lat, lng) in enumerate(sampled_points):
            try:
                print(f"\nPoint {i+1}/{len(sampled_points)}: {lat:.6f}, {lng:.6f}")
                
                # Google Places API
                try:
                    google_stations = search_ev_stations_nearby(lat, lng)
                    print(f"Found {len(google_stations)} Google stations")
                    all_stations.extend(google_stations)
                except Exception as e:
                    print(f"Google Places search failed: {str(e)}")
                
                # OpenChargeMap API
                try:
                    ocm_stations = get_charging_stations_near_point(lat, lng)
                    print(f"Found {len(ocm_stations)} OCM stations")
                    
                    processed_ocm_stations = []
                    for s in ocm_stations:
                        try:
                            station_data = {
                                'isOCM': True,
                                'place_id': str(s.get('ID', '')),
                                'name': s.get('AddressInfo', {}).get('Title', 'Unknown Station'),
                                'vicinity': s.get('AddressInfo', {}).get('AddressLine1', ''),
                                'geometry': {
                                    'location': {
                                        'lat': s.get('AddressInfo', {}).get('Latitude'),
                                        'lng': s.get('AddressInfo', {}).get('Longitude')
                                    }
                                },
                                'connectors': [],
                                'power_kw': None
                            }
                            
                            # Process connections if available
                            connections = s.get('Connections', [])
                            if connections:
                                station_data['connectors'] = [
                                    f"{c.get('ConnectionType', {}).get('Title', 'Unknown')} ({c.get('PowerKW', '?')}kW)"
                                    for c in connections
                                ]
                                station_data['power_kw'] = max(
                                    [float(c['PowerKW']) 
                                    for c in connections 
                                    if c.get('PowerKW') is not None],
                                    default=None
                                )
                            
                            processed_ocm_stations.append(station_data)
                            
                        except Exception as e:
                            print(f"Error processing OCM station: {str(e)}")
                            continue
                    
                    all_stations.extend(processed_ocm_stations)
                    
                except Exception as e:
                    print(f"OpenChargeMap search failed: {str(e)}")
                
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                print(f"Error processing point {i+1}: {str(e)}")
                continue

        # ===== 5. Deduplicate Stations =====
        print("\nProcessing stations...")
        unique_stations = []
        seen_ids = set()
        
        for station in all_stations:
            try:
                station_id = station.get('place_id') or str(station.get('ID'))
                if not station_id or station_id in seen_ids:
                    continue
                    
                seen_ids.add(station_id)
                
                # Standardize station format
                standardized = {
                    'place_id': station_id,
                    'name': station.get('name', station.get('AddressInfo', {}).get('Title')),
                    'vicinity': station.get('vicinity', station.get('AddressInfo', {}).get('AddressLine1')),
                    'rating': station.get('rating'),
                    'geometry': station.get('geometry', {
                        'location': {
                            'lat': station.get('lat') or station['AddressInfo']['Latitude'],
                            'lng': station.get('lng') or station['AddressInfo']['Longitude']
                        }
                    }),
                    'connectors': station.get('connectors', []),
                    'power_kw': station.get('power_kw'),
                    'isOCM': station.get('isOCM', False)
                }
                unique_stations.append(standardized)
                
            except Exception as e:
                print(f"Error processing station: {str(e)}")
                continue

        print(f"Found {len(unique_stations)} unique stations")

        # ===== 6. Calculate Minimum Stops =====
        print("\nCalculating recommended stops...")
        try:
            recommended_stations, stops_required = find_minimum_stops(
                route_polyline,
                unique_stations,
                float(data['current_range_km']),
                float(data['max_range_km'])
            )
            print(f"Recommended {stops_required} stops")
        except Exception as e:
            print(f"Error calculating stops: {str(e)}")
            recommended_stations = []
            stops_required = -1

        # ===== 7. Generate Waypoints =====
        print("\nGenerating waypoints...")
        try:
            osrm_route = [[p[1], p[0]] for p in route_polyline]  # Convert to OSRM format
            waypoints = generate_enhanced_waypoints(
                osrm_route,
                float(data['current_range_km'])
            )
            print(f"Generated {len(waypoints)} waypoints")
        except Exception as e:
            print(f"Error generating waypoints: {str(e)}")
            waypoints = []

        # ===== 8. Prepare Response =====
        response = {
            "stations": unique_stations,
            "recommended_stations": recommended_stations,
            "stops_required": stops_required,
            "polyline": route_polyline,
            "waypoints": waypoints if waypoints else [],  # Ensure this is always an array
            "total_route_distance_km": total_distance_km,
            "status": "success"
        }

        print("\n=== Request Completed Successfully ===")
        return jsonify(response)

    except Exception as e:
        print("\n!!! CRITICAL ERROR !!!")
        print(str(e))
        import traceback
        traceback.print_exc()
        
        return jsonify({
            "error": "An unexpected error occurred",
            "details": str(e),
            "status": "error"
        }), 500
@app.route('/ev_range', methods=['POST'])
def calculate_ev_range():
    """Calculate usable range based on EV model and battery percentage"""
    data = request.get_json()
    make = data.get('make')
    model = data.get('model')
    battery_percent = data.get('battery_percent', 100)
    
    max_range = get_ev_max_range(make, model)
    if not max_range:
        return jsonify({"error": "EV model not found"}), 404

    SAFETY_BUFFER_PERCENT = 20
    usable_percent = max(0, battery_percent - SAFETY_BUFFER_PERCENT)
    current_range_km = (usable_percent / 100.0) * max_range

    return jsonify({
        "max_range_km": max_range,
        "usable_range_km": current_range_km,
        "safety_buffer_percent": SAFETY_BUFFER_PERCENT
    })


@app.route('/ai_assistant', methods=['POST'])
def ai_assistant():
    try:
        data = request.get_json()
        user_message = data.get('message')
        ev_context = data.get('ev_context', {})
        is_voice = data.get('is_voice', False)
        
        system_prompt = f"""You are Zyntra AI, an expert assistant for EV charging. 
        Current EV Context:
        - Model: {ev_context.get('model', 'Unknown')}
        - Battery: {ev_context.get('battery', 0)}%
        
        Provide concise, helpful responses about:
        - Finding charging stations
        - Booking charging slots
        - Trip planning
        - Battery range calculations
        - Payment/wallet questions
        
        For voice responses, keep answers brief (1-2 sentences max), female voice.
        For text responses, you can provide more detailed answers.
        
        For off-topic requests, respond: "I specialize in EV charging assistance. How can I help with your charging needs?"
        """
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,
            max_tokens=200 if is_voice else 500
        )
        
        return jsonify({
            "response": response.choices[0].message.content,
            "is_voice_optimized": is_voice
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
# Add this to your existing backend code (near the other API endpoints)

@app.route('/station_details', methods=['GET'])
def get_station_details():
    """Get detailed information about a charging station from OpenChargeMap"""
    try:
        latitude = request.args.get('lat', type=float)
        longitude = request.args.get('lng', type=float)
        station_id = request.args.get('id')  # Optional station ID
        
        if latitude is None or longitude is None:
            return jsonify({"error": "Latitude and longitude parameters are required"}), 400

        # Known plug type mappings
        plug_type_map = {
            1: "Type 1 (SAE J1772)",
            2: "Type 2 (Mennekes)",
            3: "CHAdeMO",
            25: "CCS Combo 2 (IEC 62196)",
            33: "CCS Combo 1 (SAE Combo)",
            1036: "AC Type 2 (Fast)",
        }

        url = "https://api.openchargemap.io/v3/poi/"
        params = {
            "output": "json",
            "latitude": latitude,
            "longitude": longitude,
            "distance": 500,
            "distanceunit": "KM",
            "maxresults": 1,
            "compact": "false",
            "verbose": "false",
            "key": OCM_API_KEY
        }

        if station_id:
            params = {
                "output": "json",
                "chargepointid": station_id,
                "compact": "false",
                "key": OCM_API_KEY_1
            }

        response = requests.get(url, params=params)
        data = response.json()

        if not data:
            return jsonify({"error": "No charging station found at this location"}), 404

        station = data[0] if isinstance(data, list) else data
        info = station.get("AddressInfo", {})
        connections = station.get("Connections", [])
        
        connection_details = []
        for conn in connections:
            conn_type_id = conn.get("ConnectionTypeID")
            connection_details.append({
                "type": plug_type_map.get(conn_type_id, f"Type ID {conn_type_id}"),
                "power_kw": conn.get("PowerKW"),
                "voltage": conn.get("Voltage"),
                "amps": conn.get("Amps"),
                "level": conn.get("Level", {}).get("Title"),
                "current": conn.get("CurrentType", {}).get("Title")
            })

        station_data = {
            "id": station.get("ID"),
            "name": info.get("Title", "Unnamed Station"),
            "address": {
                "line1": info.get("AddressLine1"),
                "line2": info.get("AddressLine2"),
                "town": info.get("Town"),
                "state": info.get("StateOrProvince"),
                "postcode": info.get("Postcode"),
                "country": info.get("Country", {}).get("Title")
            },
            "location": {
                "lat": info.get("Latitude"),
                "lng": info.get("Longitude")
            },
            "operator": station.get("OperatorInfo", {}).get("Title"),
            "status": station.get("StatusType", {}).get("Title"),
            "connections": connection_details,
            "pricing": station.get("UsageCost"),
            "usage_type": station.get("UsageType", {}).get("Title"),
            "number_of_points": station.get("NumberOfPoints"),
            "date_created": info.get("DateCreated"),
            "date_last_verified": info.get("DateLastVerified"),
            "data_provider": station.get("DataProvider", {}).get("Title")
        }

        return jsonify(station_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
import heapq
import logging

import heapq
import logging
import datetime
from flask import jsonify, request
# You might need a fuzzy string matching library for better name comparison
# pip install thefuzz
from thefuzz import fuzz 

# --- New Helper Function: Deduplicate Stations ---
def deduplicate_stations(stations, distance_threshold_km=0.2, name_similarity_threshold=80):
    """Deduplicates a list of charging stations from multiple sources."""
    if not stations:
        return []
    
    deduplicated = []
    processed_indices = set()

    for i in range(len(stations)):
        if i in processed_indices:
            continue

        try:
            base_station = stations[i]
            if not base_station or 'location' not in base_station:
                continue
                
            duplicates = [base_station]
            processed_indices.add(i)

            for j in range(i + 1, len(stations)):
                if j in processed_indices:
                    continue
                
                try:
                    compare_station = stations[j]
                    if not compare_station or 'location' not in compare_station:
                        continue

                    distance = haversine_km(
                        (base_station['location']['lat'], base_station['location']['lng']),
                        (compare_station['location']['lat'], compare_station['location']['lng'])
                    )

                    if distance < distance_threshold_km:
                        name_similarity = fuzz.token_sort_ratio(
                            base_station.get('name', ''),
                            compare_station.get('name', '')
                        )
                        if name_similarity > name_similarity_threshold:
                            duplicates.append(compare_station)
                            processed_indices.add(j)
                except Exception as e:
                    logging.error(f"Error comparing station {j}: {e}")
                    continue

            # Merge duplicates
            merged_station = merge_station_data(duplicates)
            if merged_station:
                deduplicated.append(merged_station)
                
        except Exception as e:
            logging.error(f"Error processing base station {i}: {e}")
            continue

    return deduplicated

def merge_station_data(duplicates):
    """Helper function to merge station data"""
    if not duplicates:
        return None
        
    # Initialize with first station
    merged = duplicates[0].copy()
    
    # Merge data from other stations
    for station in duplicates[1:]:
        try:
            # Prefer Google data for certain fields
            if station.get('source') == 'google':
                for field in ['name', 'address', 'rating', 'user_ratings_total', 'opening_hours']:
                    if field in station and station[field]:
                        merged[field] = station[field]
            
            # Merge connector information
            if 'connectors' in station and station['connectors']:
                if 'connectors' not in merged:
                    merged['connectors'] = []
                merged['connectors'].extend(
                    conn for conn in station['connectors'] 
                    if conn not in merged['connectors']
                )
                
            # Keep highest power connector
            if 'fastest_connector' in station and station['fastest_connector']['power_kw'] > merged.get('fastest_connector', {}).get('power_kw', 0):
                merged['fastest_connector'] = station['fastest_connector']
                
        except Exception as e:
            logging.error(f"Error merging station data: {e}")
            continue
            
    merged['source'] = 'merged'
    return merged

# --- Improved Main Route ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5501, debug=True)