import React, { useState } from 'react';
import { Droplet, Thermometer, Cloud, Wind, Sun, Sprout, Zap, Camera, Activity, Database, Brain, Smartphone, AlertTriangle, TrendingUp, Wifi } from 'lucide-react';

const SmartAgricultureSystem = () => {
  const [activeTab, setActiveTab] = useState('sensors');

  const sensors = [
    {
      name: 'Soil Moisture Sensor',
      icon: Droplet,
      description: 'Measures volumetric water content in soil',
      specs: 'Range: 0-100% VWC, Accuracy: ±3%',
      dataType: 'soil_moisture (%)',
      typical: '20-80%',
      critical: '<20% or >80%',
      cost: '$15-50',
      placement: 'Root zone depth (6-12 inches)'
    },
    {
      name: 'Temperature Sensor',
      icon: Thermometer,
      description: 'Monitors ambient and soil temperature',
      specs: 'Range: -40°C to 80°C, Accuracy: ±0.5°C',
      dataType: 'temperature (°C)',
      typical: '15-35°C',
      critical: '<5°C or >40°C',
      cost: '$10-30',
      placement: 'Air (1.5m height) & Soil (2-4 inches)'
    },
    {
      name: 'Humidity Sensor',
      icon: Cloud,
      description: 'Measures relative humidity',
      specs: 'Range: 0-100% RH, Accuracy: ±2%',
      dataType: 'humidity (%)',
      typical: '40-70%',
      critical: '<30% or >85%',
      cost: '$8-25',
      placement: 'Canopy level (1-2m height)'
    },
    {
      name: 'pH Sensor',
      icon: Activity,
      description: 'Monitors soil acidity/alkalinity',
      specs: 'Range: 0-14 pH, Accuracy: ±0.1',
      dataType: 'pH',
      typical: '6.0-7.5',
      critical: '<5.5 or >8.0',
      cost: '$50-150',
      placement: 'Root zone depth'
    },
    {
      name: 'NPK Sensor',
      icon: Zap,
      description: 'Measures Nitrogen, Phosphorus, Potassium',
      specs: 'Range: 0-1999 mg/kg',
      dataType: 'N, P, K (mg/kg)',
      typical: 'N:50-200, P:30-100, K:100-300',
      critical: 'N<30, P<20, K<50',
      cost: '$100-300',
      placement: 'Root zone depth'
    },
    {
      name: 'Light Sensor',
      icon: Sun,
      description: 'Measures light intensity (PAR)',
      specs: 'Range: 0-2000 μmol/m²/s',
      dataType: 'light_intensity (lux)',
      typical: '400-700 μmol/m²/s',
      critical: '<200 μmol/m²/s',
      cost: '$30-80',
      placement: 'Canopy level'
    },
    {
      name: 'Weather Station',
      icon: Wind,
      description: 'Wind speed, rainfall, pressure',
      specs: 'Multi-parameter monitoring',
      dataType: 'wind_speed, rainfall, pressure',
      typical: 'Variable by location',
      critical: 'Wind >50 km/h, Heavy rain',
      cost: '$200-500',
      placement: 'Open area, 3-10m height'
    },
    {
      name: 'Camera/Vision System',
      icon: Camera,
      description: 'Plant health, pest detection, growth monitoring',
      specs: '5MP+, IP67, night vision',
      dataType: 'images, video',
      typical: 'Healthy green foliage',
      critical: 'Disease, pest infestation',
      cost: '$50-200',
      placement: 'Multiple angles, 1-2m height'
    }
  ];

  const aiModels = [
    {
      name: 'Crop Yield Prediction',
      type: 'Regression',
      algorithm: 'Random Forest / Gradient Boosting',
      inputs: ['Soil moisture', 'Temperature', 'Humidity', 'pH', 'NPK levels', 'Historical yield', 'Weather data', 'Growth stage'],
      outputs: ['Expected yield (kg/hectare)', 'Confidence interval', 'Optimal harvest date'],
      accuracy: '85-92%',
      description: 'Predicts final crop yield based on current conditions and historical data'
    },
    {
      name: 'Disease Detection',
      type: 'Image Classification',
      algorithm: 'CNN (MobileNetV2 / ResNet)',
      inputs: ['Leaf images', 'Plant images', 'Environmental data'],
      outputs: ['Disease type', 'Severity level', 'Treatment recommendation'],
      accuracy: '90-96%',
      description: 'Identifies plant diseases from visual symptoms'
    },
    {
      name: 'Irrigation Optimizer',
      type: 'Reinforcement Learning',
      algorithm: 'Deep Q-Network (DQN)',
      inputs: ['Soil moisture', 'Weather forecast', 'Crop type', 'Growth stage', 'Evapotranspiration'],
      outputs: ['Irrigation schedule', 'Water volume', 'Timing'],
      accuracy: '30% water savings',
      description: 'Optimizes irrigation timing and volume to minimize water waste'
    },
    {
      name: 'Pest Detection',
      type: 'Object Detection',
      algorithm: 'YOLO / SSD',
      inputs: ['Camera feeds', 'Trap images', 'Temperature', 'Humidity'],
      outputs: ['Pest species', 'Population density', 'Alert threshold'],
      accuracy: '88-94%',
      description: 'Detects and counts pest populations in real-time'
    }
  ];

  const dataFlow = {
    layers: [
      {
        name: 'Sensor Layer',
        items: ['IoT Sensors', 'Cameras', 'Weather Station', 'Drones'],
        color: 'bg-blue-100 border-blue-300'
      },
      {
        name: 'Communication Layer',
        items: ['LoRaWAN', 'WiFi', '4G/5G', 'Zigbee'],
        color: 'bg-green-100 border-green-300'
      },
      {
        name: 'Edge Processing',
        items: ['Data Aggregation', 'Initial Filtering', 'Anomaly Detection', 'Alert Generation'],
        color: 'bg-purple-100 border-purple-300'
      },
      {
        name: 'Cloud Platform',
        items: ['Data Storage', 'AI Model Training', 'Historical Analysis', 'API Services'],
        color: 'bg-orange-100 border-orange-300'
      },
      {
        name: 'AI/ML Layer',
        items: ['Yield Prediction', 'Disease Detection', 'Irrigation Optimization', 'Pest Monitoring'],
        color: 'bg-pink-100 border-pink-300'
      },
      {
        name: 'Application Layer',
        items: ['Mobile App', 'Web Dashboard', 'SMS Alerts', 'Automation Control'],
        color: 'bg-yellow-100 border-yellow-300'
      }
    ]
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 to-blue-50 p-6">
      <div className="max-w-7xl mx-auto">
        <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
          <div className="flex items-center gap-3 mb-4">
            <Sprout className="w-10 h-10 text-green-600" />
            <div>
              <h1 className="text-3xl font-bold text-gray-800">Smart Agriculture System</h1>
              <p className="text-gray-600">IoT + AI for Precision Farming</p>
            </div>
          </div>
          
          <div className="flex gap-2 mt-4 flex-wrap">
            <button
              onClick={() => setActiveTab('sensors')}
              className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                activeTab === 'sensors'
                  ? 'bg-green-600 text-white'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              IoT Sensors
            </button>
            <button
              onClick={() => setActiveTab('ai')}
              className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                activeTab === 'ai'
                  ? 'bg-green-600 text-white'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              AI Models
            </button>
            <button
              onClick={() => setActiveTab('dataflow')}
              className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                activeTab === 'dataflow'
                  ? 'bg-green-600 text-white'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              Data Flow
            </button>
            <button
              onClick={() => setActiveTab('implementation')}
              className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                activeTab === 'implementation'
                  ? 'bg-green-600 text-white'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              Implementation
            </button>
          </div>
        </div>

        {activeTab === 'sensors' && (
          <div className="space-y-4">
            <div className="bg-white rounded-lg shadow-lg p-6">
              <h2 className="text-2xl font-bold text-gray-800 mb-4">IoT Sensor Network</h2>
              <p className="text-gray-600 mb-6">
                Essential sensors for comprehensive farm monitoring and data collection
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {sensors.map((sensor, index) => {
                const Icon = sensor.icon;
                return (
                  <div key={index} className="bg-white rounded-lg shadow-md p-5 hover:shadow-xl transition-shadow">
                    <div className="flex items-start gap-3 mb-3">
                      <div className="p-2 bg-green-100 rounded-lg">
                        <Icon className="w-6 h-6 text-green-600" />
                      </div>
                      <div className="flex-1">
                        <h3 className="text-lg font-bold text-gray-800">{sensor.name}</h3>
                        <p className="text-sm text-gray-600">{sensor.description}</p>
                      </div>
                    </div>
                    
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-gray-600">Specifications:</span>
                        <span className="text-gray-800 font-medium">{sensor.specs}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Data Type:</span>
                        <span className="text-gray-800 font-mono text-xs">{sensor.dataType}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Typical Range:</span>
                        <span className="text-green-600 font-medium">{sensor.typical}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Critical Levels:</span>
                        <span className="text-red-600 font-medium">{sensor.critical}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Cost:</span>
                        <span className="text-blue-600 font-medium">{sensor.cost}</span>
                      </div>
                      <div className="pt-2 border-t">
                        <span className="text-gray-600">Placement: </span>
                        <span className="text-gray-800">{sensor.placement}</span>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {activeTab === 'ai' && (
          <div className="space-y-4">
            <div className="bg-white rounded-lg shadow-lg p-6">
              <h2 className="text-2xl font-bold text-gray-800 mb-4">AI/ML Models</h2>
              <p className="text-gray-600 mb-6">
                Intelligent algorithms for prediction, detection, and optimization
              </p>
            </div>

            {aiModels.map((model, index) => (
              <div key={index} className="bg-white rounded-lg shadow-md p-6 hover:shadow-xl transition-shadow">
                <div className="flex items-start gap-3 mb-4">
                  <Brain className="w-8 h-8 text-purple-600 flex-shrink-0" />
                  <div className="flex-1">
                    <h3 className="text-xl font-bold text-gray-800 mb-1">{model.name}</h3>
                    <p className="text-gray-600 mb-3">{model.description}</p>
                    
                    <div className="flex gap-4 mb-4 flex-wrap">
                      <span className="px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-sm font-medium">
                        {model.type}
                      </span>
                      <span className="px-3 py-1 bg-purple-100 text-purple-700 rounded-full text-sm font-medium">
                        {model.algorithm}
                      </span>
                      <span className="px-3 py-1 bg-green-100 text-green-700 rounded-full text-sm font-medium">
                        Accuracy: {model.accuracy}
                      </span>
                    </div>

                    <div className="grid md:grid-cols-2 gap-4">
                      <div className="border border-gray-200 rounded-lg p-3">
                        <h4 className="font-semibold text-gray-700 mb-2 flex items-center gap-2">
                          <Database className="w-4 h-4" />
                          Input Features
                        </h4>
                        <ul className="space-y-1">
                          {model.inputs.map((input, i) => (
                            <li key={i} className="text-sm text-gray-600 flex items-center gap-2">
                              <span className="w-1.5 h-1.5 bg-blue-500 rounded-full"></span>
                              {input}
                            </li>
                          ))}
                        </ul>
                      </div>

                      <div className="border border-gray-200 rounded-lg p-3">
                        <h4 className="font-semibold text-gray-700 mb-2 flex items-center gap-2">
                          <TrendingUp className="w-4 h-4" />
                          Output Predictions
                        </h4>
                        <ul className="space-y-1">
                          {model.outputs.map((output, i) => (
                            <li key={i} className="text-sm text-gray-600 flex items-center gap-2">
                              <span className="w-1.5 h-1.5 bg-green-500 rounded-full"></span>
                              {output}
                            </li>
                          ))}
                        </ul>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            ))}

            <div className="bg-gradient-to-r from-green-50 to-blue-50 rounded-lg shadow-md p-6">
              <h3 className="text-xl font-bold text-gray-800 mb-4">Crop Yield Prediction Model Architecture</h3>
              
              <div className="space-y-4">
                <div className="bg-white rounded-lg p-4">
                  <h4 className="font-semibold text-gray-700 mb-2">Model Type: Ensemble (Random Forest + XGBoost)</h4>
                  <p className="text-sm text-gray-600 mb-3">
                    Combines multiple decision trees to improve prediction accuracy and reduce overfitting
                  </p>
                  
                  <div className="grid md:grid-cols-3 gap-3">
                    <div className="border border-gray-200 rounded p-3">
                      <div className="text-xs text-gray-500 mb-1">Training Data</div>
                      <div className="font-semibold text-gray-800">5+ years historical data</div>
                      <div className="text-xs text-gray-600">Weather, soil, yield records</div>
                    </div>
                    <div className="border border-gray-200 rounded p-3">
                      <div className="text-xs text-gray-500 mb-1">Features</div>
                      <div className="font-semibold text-gray-800">50+ parameters</div>
                      <div className="text-xs text-gray-600">Sensor data + derivatives</div>
                    </div>
                    <div className="border border-gray-200 rounded p-3">
                      <div className="text-xs text-gray-500 mb-1">Update Frequency</div>
                      <div className="font-semibold text-gray-800">Daily predictions</div>
                      <div className="text-xs text-gray-600">Real-time adjustment</div>
                    </div>
                  </div>
                </div>

                <div className="bg-white rounded-lg p-4">
                  <h4 className="font-semibold text-gray-700 mb-3">Feature Engineering</h4>
                  <div className="grid md:grid-cols-2 gap-3 text-sm">
                    <div>
                      <span className="font-medium text-gray-700">Time-based features:</span>
                      <ul className="list-disc list-inside text-gray-600 mt-1 ml-2">
                        <li>Days since planting</li>
                        <li>Growth degree days (GDD)</li>
                        <li>Seasonal indicators</li>
                      </ul>
                    </div>
                    <div>
                      <span className="font-medium text-gray-700">Derived features:</span>
                      <ul className="list-disc list-inside text-gray-600 mt-1 ml-2">
                        <li>Moisture stress index</li>
                        <li>Temperature stress days</li>
                        <li>NPK balance ratio</li>
                      </ul>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'dataflow' && (
          <div className="space-y-4">
            <div className="bg-white rounded-lg shadow-lg p-6">
              <h2 className="text-2xl font-bold text-gray-800 mb-4">System Data Flow Architecture</h2>
              <p className="text-gray-600 mb-6">
                End-to-end data pipeline from sensors to actionable insights
              </p>
            </div>

            <div className="bg-white rounded-lg shadow-lg p-6">
              {dataFlow.layers.map((layer, index) => (
                <div key={index} className="mb-6 last:mb-0">
                  <div className="flex items-center gap-3 mb-3">
                    <div className="w-8 h-8 bg-green-600 text-white rounded-full flex items-center justify-center font-bold">
                      {index + 1}
                    </div>
                    <h3 className="text-lg font-bold text-gray-800">{layer.name}</h3>
                  </div>
                  
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3 ml-11">
                    {layer.items.map((item, i) => (
                      <div
                        key={i}
                        className={`${layer.color} border-2 rounded-lg p-3 text-center font-medium text-gray-700`}
                      >
                        {item}
                      </div>
                    ))}
                  </div>

                  {index < dataFlow.layers.length - 1 && (
                    <div className="flex justify-center my-3">
                      <div className="text-green-600 text-2xl">↓</div>
                    </div>
                  )}
                </div>
              ))}
            </div>

            <div className="bg-white rounded-lg shadow-lg p-6">
              <h3 className="text-xl font-bold text-gray-800 mb-4">Real-Time Data Processing Pipeline</h3>
              
              <div className="space-y-4">
                <div className="flex items-start gap-3">
                  <div className="w-8 h-8 bg-blue-500 text-white rounded-full flex items-center justify-center flex-shrink-0 font-bold">
                    1
                  </div>
                  <div className="flex-1">
                    <h4 className="font-semibold text-gray-800 mb-1">Data Collection (Every 15 minutes)</h4>
                    <p className="text-sm text-gray-600">Sensors collect environmental data and transmit to edge gateway via LoRaWAN/WiFi</p>
                    <div className="mt-2 p-2 bg-gray-50 rounded text-xs font-mono overflow-x-auto">
                      sensor_data = temp: 25.3, moisture: 45.2, pH: 6.8
                    </div>
                  </div>
                </div>

                <div className="flex items-start gap-3">
                  <div className="w-8 h-8 bg-green-500 text-white rounded-full flex items-center justify-center flex-shrink-0 font-bold">
                    2
                  </div>
                  <div className="flex-1">
                    <h4 className="font-semibold text-gray-800 mb-1">Edge Processing (1-5 seconds)</h4>
                    <p className="text-sm text-gray-600">Local gateway validates data, detects anomalies, and triggers immediate alerts</p>
                    <div className="mt-2 p-2 bg-gray-50 rounded text-xs font-mono">
                      if moisture under 20%: trigger irrigation alert
                    </div>
                  </div>
                </div>

                <div className="flex items-start gap-3">
                  <div className="w-8 h-8 bg-purple-500 text-white rounded-full flex items-center justify-center flex-shrink-0 font-bold">
                    3
                  </div>
                  <div className="flex-1">
                    <h4 className="font-semibold text-gray-800 mb-1">Cloud Storage (Real-time)</h4>
                    <p className="text-sm text-gray-600">Data synced to cloud database for long-term storage and batch processing</p>
                    <div className="mt-2 p-2 bg-gray-50 rounded text-xs font-mono">
                      TimescaleDB: INSERT sensor readings
                    </div>
                  </div>
                </div>

                <div className="flex items-start gap-3">
                  <div className="w-8 h-8 bg-pink-500 text-white rounded-full flex items-center justify-center flex-shrink-0 font-bold">
                    4
                  </div>
                  <div className="flex-1">
                    <h4 className="font-semibold text-gray-800 mb-1">AI Model Inference (Hourly/Daily)</h4>
                    <p className="text-sm text-gray-600">ML models process historical and current data to generate predictions</p>
                    <div className="mt-2 p-2 bg-gray-50 rounded text-xs font-mono">
                      yield = model.predict(sensor_data, weather)
                    </div>
                  </div>
                </div>

                <div className="flex items-start gap-3">
                  <div className="w-8 h-8 bg-orange-500 text-white rounded-full flex items-center justify-center flex-shrink-0 font-bold">
                    5
                  </div>
                  <div className="flex-1">
                    <h4 className="font-semibold text-gray-800 mb-1">Action & Notification (Instant)</h4>
                    <p className="text-sm text-gray-600">Results delivered via mobile app, SMS, or automated control</p>
                    <div className="mt-2 p-2 bg-gray-50 rounded text-xs font-mono">
                      SMS: Low soil moisture detected in Zone A
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-lg shadow-lg p-6">
              <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
                <Wifi className="w-6 h-6 text-green-600" />
                Communication Technologies
              </h3>
              
              <div className="grid md:grid-cols-2 gap-4">
                <div className="border border-gray-200 rounded-lg p-4">
                  <h4 className="font-semibold text-gray-700 mb-2">LoRaWAN (Long Range)</h4>
                  <ul className="text-sm text-gray-600 space-y-1">
                    <li>• Range: Up to 10-15 km in rural areas</li>
                    <li>• Power: Battery lasts 2-5 years</li>
                    <li>• Best for: Remote sensor nodes</li>
                    <li>• Data rate: 0.3-50 kbps</li>
                  </ul>
                </div>

                <div className="border border-gray-200 rounded-lg p-4">
                  <h4 className="font-semibold text-gray-700 mb-2">WiFi / 4G/5G</h4>
                  <ul className="text-sm text-gray-600 space-y-1">
                    <li>• Range: 50-100m (WiFi), Wide (Cellular)</li>
                    <li>• Power: Higher consumption</li>
                    <li>• Best for: Cameras, high-bandwidth</li>
                    <li>• Data rate: 1-100+ Mbps</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'implementation' && (
          <div className="space-y-4">
            <div className="bg-white rounded-lg shadow-lg p-6">
              <h2 className="text-2xl font-bold text-gray-800 mb-4">Implementation Guide</h2>
              <p className="text-gray-600 mb-6">
                Step-by-step deployment for a 10-hectare farm
              </p>

              <div className="space-y-6">
                <div className="border-l-4 border-green-500 pl-4">
                  <h3 className="font-bold text-lg text-gray-800 mb-2">Phase 1: Planning & Design</h3>
                  <ul className="space-y-2 text-gray-700">
                    <li>• Conduct site survey and network coverage analysis</li>
                    <li>• Identify sensor placement zones based on crop layout</li>
                    <li>• Calculate sensor density (2-3 nodes per hectare)</li>
                    <li>• Design power supply (solar panels for remote sensors)</li>
                  </ul>
                </div>

                <div className="border-l-4 border-blue-500 pl-4">
                  <h3 className="font-bold text-lg text-gray-800 mb-2">Phase 2: Hardware Setup</h3>
                  <ul className="space-y-2 text-gray-700">
                    <li>• Install 20-30 sensor nodes across 10 hectares</li>
                    <li>• Deploy edge gateway (Raspberry Pi 4 or industrial PC)</li>
                    <li>• Setup weather station at central location</li>
                    <li>• Install cameras at strategic monitoring points</li>
                  </ul>
                </div>

                <div className="border-l-4 border-purple-500 pl-4">
                  <h3 className="font-bold text-lg text-gray-800 mb-2">Phase 3: Software Configuration</h3>
                  <ul className="space-y-2 text-gray-700">
                    <li>• Deploy IoT platform (AWS IoT Core / Azure IoT Hub)</li>
                    <li>• Configure TimescaleDB for time-series data storage</li>
                    <li>• Setup TensorFlow Lite models on edge devices</li>
                    <li>• Develop mobile app for farmer interface</li>
                  </ul>
                </div>

                <div className="border-l-4 border-orange-500 pl-4">
                  <h3 className="font-bold text-lg text-gray-800 mb-2">Phase 4: Model Training</h3>
                  <ul className="space-y-2 text-gray-700">
                    <li>• Collect baseline data for 2-4 weeks</li>
                    <li>• Train crop yield prediction model on historical data</li>
                    <li>• Fine-tune disease detection model with local images</li>
                    <li>• Calibrate irrigation optimizer for specific crops</li>
                  </ul>
                </div>

                <div className="border-l-4 border-pink-500 pl-4">
                  <h3 className="font-bold text-lg text-gray-800 mb-2">Phase 5: Testing & Deployment</h3>
                  <ul className="space-y-2 text-gray-700">
                    <li>• Run pilot test on 1-2 hectares</li>
                    <li>• Validate prediction accuracy against ground truth</li>
                    <li>• Adjust thresholds and alert parameters</li>
                    <li>• Roll out to full farm after successful pilot</li>
                  </ul>
                </div>
              </div>

              <div className="bg-gradient-to-r from-blue-50 to-green-50 rounded-lg p-6 mt-6">
                <h3 className="font-bold text-xl text-gray-800 mb-4">Cost Breakdown (10 Hectares)</h3>
                
                <div className="grid md:grid-cols-2 gap-4">
                  <div className="bg-white rounded-lg p-4">
                    <h4 className="font-semibold text-gray-700 mb-3">Initial Investment</h4>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span>Sensors (25 nodes × $200)</span>
                        <span className="font-semibold">$5,000</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Edge Gateway Hardware</span>
                        <span className="font-semibold">$500</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Cameras (5 units)</span>
                        <span className="font-semibold">$750</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Weather Station</span>
                        <span className="font-semibold">$400</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Installation & Setup</span>
                        <span className="font-semibold">$1,500</span>
                      </div>
                      <div className="border-t pt-2 flex justify-between font-bold">
                        <span>Total Hardware</span>
                        <span className="text-green-600">$8,150</span>
                      </div>
                    </div>
                  </div>

                  <div className="bg-white rounded-lg p-4">
                    <h4 className="font-semibold text-gray-700 mb-3">Annual Operating Costs</h4>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span>Cloud Services (AWS/Azure)</span>
                        <span className="font-semibold">$600/year</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Cellular Data (4G)</span>
                        <span className="font-semibold">$240/year</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Maintenance & Calibration</span>
                        <span className="font-semibold">$500/year</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Software Updates</span>
                        <span className="font-semibold">$300/year</span>
                      </div>
                      <div className="border-t pt-2 flex justify-between font-bold">
                        <span>Annual Total</span>
                        <span className="text-blue-600">$1,640/year</span>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="mt-4 bg-white rounded-lg p-4">
                  <h4 className="font-semibold text-gray-700 mb-2">Expected ROI</h4>
                  <ul className="text-sm text-gray-700 space-y-1">
                    <li>✓ 20-30% increase in crop yield through optimization</li>
                    <li>✓ 30-40% reduction in water usage</li>
                    <li>✓ 25% reduction in fertilizer waste</li>
                    <li>✓ Early disease detection saves 10-15% of crops</li>
                    <li className="font-semibold text-green-600 pt-2">Payback period: 12-18 months</li>
                  </ul>
                </div>
              </div>

              <div className="bg-white rounded-lg shadow-lg p-6 mt-6">
                <h3 className="font-bold text-xl text-gray-800 mb-4">System Architecture Diagram</h3>
                
                <div className="space-y-3">
                  <div className="flex items-center gap-3">
                    <div className="w-32 bg-blue-100 border-2 border-blue-400 rounded-lg p-3 text-center">
                      <div className="font-semibold text-blue-800">Field Sensors</div>
                      <div className="text-xs text-blue-600">IoT Layer</div>
                    </div>
                    <div className="flex-1 border-t-2 border-dashed border-gray-400"></div>
                    <div className="text-gray-500">LoRaWAN/WiFi</div>
                    <div className="flex-1 border-t-2 border-dashed border-gray-400"></div>
                    <div className="w-32 bg-green-100 border-2 border-green-400 rounded-lg p-3 text-center">
                      <div className="font-semibold text-green-800">Edge Gateway</div>
                      <div className="text-xs text-green-600">Raspberry Pi</div>
                    </div>
                  </div>

                  <div className="flex justify-center">
                    <div className="text-green-600 text-2xl">↕</div>
                  </div>

                  <div className="flex items-center justify-center">
                    <div className="w-48 bg-orange-100 border-2 border-orange-400 rounded-lg p-3 text-center">
                      <div className="font-semibold text-orange-800">Cloud Platform</div>
                      <div className="text-xs text-orange-600">AWS IoT / Azure</div>
                    </div>
                  </div>

                  <div className="flex justify-center">
                    <div className="text-orange-600 text-2xl">↕</div>
                  </div>

                  <div className="flex items-center gap-3">
                    <div className="flex-1 bg-purple-100 border-2 border-purple-400 rounded-lg p-3 text-center">
                      <div className="font-semibold text-purple-800">AI Models</div>
                      <div className="text-xs text-purple-600">ML Processing</div>
                    </div>
                    <div className="flex-1 bg-pink-100 border-2 border-pink-400 rounded-lg p-3 text-center">
                      <div className="font-semibold text-pink-800">Database</div>
                      <div className="text-xs text-pink-600">TimescaleDB</div>
                    </div>
                  </div>

                  <div className="flex justify-center">
                    <div className="text-pink-600 text-2xl">↓</div>
                  </div>

                  <div className="flex items-center gap-3">
                    <div className="flex-1 bg-yellow-100 border-2 border-yellow-400 rounded-lg p-3 text-center">
                      <div className="font-semibold text-yellow-800">Mobile App</div>
                      <div className="text-xs text-yellow-600">Farmer Interface</div>
                    </div>
                    <div className="flex-1 bg-indigo-100 border-2 border-indigo-400 rounded-lg p-3 text-center">
                      <div className="font-semibold text-indigo-800">Web Dashboard</div>
                      <div className="text-xs text-indigo-600">Analytics</div>
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-lg shadow-lg p-6 mt-6">
                <h3 className="font-bold text-xl text-gray-800 mb-4 flex items-center gap-2">
                  <AlertTriangle className="w-6 h-6 text-amber-600" />
                  Key Success Factors
                </h3>
                
                <div className="grid md:grid-cols-2 gap-4">
                  <div className="space-y-3">
                    <div className="flex items-start gap-3">
                      <div className="w-6 h-6 bg-green-500 rounded-full flex items-center justify-center text-white text-sm flex-shrink-0">✓</div>
                      <div>
                        <div className="font-semibold text-gray-800">Reliable Connectivity</div>
                        <div className="text-sm text-gray-600">Ensure network coverage across entire farm</div>
                      </div>
                    </div>
                    <div className="flex items-start gap-3">
                      <div className="w-6 h-6 bg-green-500 rounded-full flex items-center justify-center text-white text-sm flex-shrink-0">✓</div>
                      <div>
                        <div className="font-semibold text-gray-800">Regular Calibration</div>
                        <div className="text-sm text-gray-600">Calibrate sensors every 3-6 months</div>
                      </div>
                    </div>
                    <div className="flex items-start gap-3">
                      <div className="w-6 h-6 bg-green-500 rounded-full flex items-center justify-center text-white text-sm flex-shrink-0">✓</div>
                      <div>
                        <div className="font-semibold text-gray-800">Data Quality</div>
                        <div className="text-sm text-gray-600">Clean and validate sensor data continuously</div>
                      </div>
                    </div>
                  </div>
                  
                  <div className="space-y-3">
                    <div className="flex items-start gap-3">
                      <div className="w-6 h-6 bg-green-500 rounded-full flex items-center justify-center text-white text-sm flex-shrink-0">✓</div>
                      <div>
                        <div className="font-semibold text-gray-800">Farmer Training</div>
                        <div className="text-sm text-gray-600">Educate farmers on system interpretation</div>
                      </div>
                    </div>
                    <div className="flex items-start gap-3">
                      <div className="w-6 h-6 bg-green-500 rounded-full flex items-center justify-center text-white text-sm flex-shrink-0">✓</div>
                      <div>
                        <div className="font-semibold text-gray-800">Model Updates</div>
                        <div className="text-sm text-gray-600">Retrain AI models with new seasonal data</div>
                      </div>
                    </div>
                    <div className="flex items-start gap-3">
                      <div className="w-6 h-6 bg-green-500 rounded-full flex items-center justify-center text-white text-sm flex-shrink-0">✓</div>
                      <div>
                        <div className="font-semibold text-gray-800">Backup Systems</div>
                        <div className="text-sm text-gray-600">Edge processing ensures operation during outages</div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-gradient-to-r from-green-100 to-blue-100 rounded-lg p-6 mt-6">
                <h3 className="font-bold text-xl text-gray-800 mb-3">Sample Code: Crop Yield Prediction Model</h3>
                <div className="bg-gray-900 rounded-lg p-4 overflow-x-auto">
                  <pre className="text-green-400 text-xs font-mono">
{`import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Load sensor data
data = pd.read_csv('sensor_readings.csv')
features = ['soil_moisture', 'temperature', 'humidity', 
            'pH', 'nitrogen', 'phosphorus', 'potassium',
            'rainfall', 'sunshine_hours', 'growth_days']
X = data[features]
y = data['yield_kg_per_hectare']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest model
model = RandomForestRegressor(
    n_estimators=100, 
    max_depth=15,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
score = model.score(X_test, y_test)
print(f'Model R² Score: {score:.3f}')

# Make prediction for current conditions
current_data = [[45.2, 25.3, 65.0, 6.8, 
                 150, 80, 250, 120, 8.5, 85]]
predicted_yield = model.predict(current_data)
print(f'Predicted Yield: {predicted_yield[0]:.1f} kg/ha')

# Convert to TensorFlow Lite for edge deployment
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('yield_model.tflite', 'wb') as f:
    f.write(tflite_model)`}
                  </pre>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default SmartAgricultureSystem;


