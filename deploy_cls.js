// --- CRITICAL PRE-PROCESSING CONSTANTS ---

// 1. Numeric Feature Order
const NUMERIC_FEATURES = ['CO AQI Value', 'Ozone AQI Value', 'NO2 AQI Value', 'PM2.5 AQI Value'];

// 2. Standard Scaler Statistics (✅ UPDATED with Python mean_ and scale_ arrays)
const SCALER_MEANS = [2.0537, 35.8034, 4.0901, 47.9602]; 
const SCALER_STDS = [3.6358, 19.3361, 8.0163, 47.8938]; 

// 3. One-Hot Encoder Categories and Order
const OHE_CATEGORIES = [
    'Brazil', 'Canada', 'China', 'France', 'Germany', 'India', 'Japan', 
    'Russian Federation', 'United Kingdom of Great Britain and Northern Ireland', 
    'United States of America', 'Other'
];

// --- MODEL SETUP ---
let clsSession = null;
// 🛑 FIX: Updated ONNX model name to match user's saved file
const CLS_MODEL_PATH = './classification_model (1).onnx'; 

// --- MAIN EXECUTION LOGIC (FIX FOR "CANNOT SET PROPERTIES OF NULL") ---
document.addEventListener('DOMContentLoaded', () => {
    // We can now safely access the element because the DOM is loaded
    const modelStatusElement = document.getElementById('model-status');

    async function loadModels() {
        try {
            if (modelStatusElement) {
                modelStatusElement.innerText = "Loading classification model...";
            }

            clsSession = await ort.InferenceSession.create(CLS_MODEL_PATH);

            if (modelStatusElement) {
                modelStatusElement.innerText = "✅ Classification model loaded. Ready for prediction.";
            }
        } catch (e) {
            if (modelStatusElement) {
                modelStatusElement.innerText = `❌ Error loading classification model. Check console for model file path/naming errors (Expected: ${CLS_MODEL_PATH}).`;
            }
            console.error("Error loading ONNX model:", e);
        }
    }
    
    loadModels();
});

// --- PRE-PROCESSING LOGIC ---
function scaleNumericFeatures(values) {
    return values.map((val, i) => (val - SCALER_MEANS[i]) / SCALER_STDS[i]);
}

function encodeCategoricalFeatures(country) {
    const encodedArray = new Array(OHE_CATEGORIES.length).fill(0);
    let optimizedCountry = country;
    if (!OHE_CATEGORIES.includes(country)) {
        optimizedCountry = 'Other';
    }
    
    const index = OHE_CATEGORIES.indexOf(optimizedCountry);
    if (index !== -1) {
        encodedArray[index] = 1;
    }
    return encodedArray;
}

function buildInputTensor(inputs) {
    const { co_aqi, ozone_aqi, no2_aqi, pm25_aqi, country } = inputs;
    
    const numericValues = [co_aqi, ozone_aqi, no2_aqi, pm25_aqi];
    const scaledNumeric = scaleNumericFeatures(numericValues);
    const encodedCategorical = encodeCategoricalFeatures(country);
    
    // Final Tensor: [4 Scaled Numeric, 11 OHE Categorical] = 15 features
    const finalInput = [...scaledNumeric, ...encodedCategorical];
    
    return new ort.Tensor('float32', finalInput, [1, finalInput.length]);
}


// --- INFERENCE LOGIC (Must be globally accessible) ---

async function predictAQI() {
    if (!clsSession) {
        alert("Model is still loading. Please wait.");
        return;
    }

    // 1. Collect inputs
    const inputs = {
        co_aqi: parseFloat(document.getElementById('co_aqi').value),
        ozone_aqi: parseFloat(document.getElementById('ozone_aqi').value),
        no2_aqi: parseFloat(document.getElementById('no2_aqi').value),
        pm25_aqi: parseFloat(document.getElementById('pm25_aqi').value),
        country: document.getElementById('country').value
    };

    // 2. Pre-process and create tensor
    const inputTensor = buildInputTensor(inputs);
    const feeds = { 'input': inputTensor };

    // 3. Run Classification Inference
    try {
        const clsResults = await clsSession.run(feeds);
        const clsLogit = clsResults.output.data[0];
        
        // Apply Sigmoid to the logit
        const probability = 1 / (1 + Math.exp(-clsLogit));
        
        // Classify: AQI Category 'Good' is the target (1)
        const isGood = probability >= 0.5;
        const aqiCategory = isGood ? "Good (or better)" : "Not Good (Moderate to Unhealthy)";
        
        document.getElementById('cls_output').innerText = `${aqiCategory} (Probability: ${probability.toFixed(4)})`;

    } catch (e) {
        document.getElementById('cls_output').innerText = "Prediction Error";
        console.error("Inference failed:", e);
    }
}
