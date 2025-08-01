// Reference: https://codepen.io/mediapipe-preview/pen/abRLMxN
import { PoseLandmarker, FilesetResolver, DrawingUtils } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";

const videoElement = document.getElementById('webcamVideo');
const canvasElement = document.getElementById('outputCanvas');
const canvasCtx = canvasElement.getContext('2d');
const startCameraButton = document.getElementById('startCameraButton');
const toggleVisualizationButton = document.getElementById('toggleVisualizationButton');
const toggleTrackingButton = document.getElementById('toggleTrackingButton');
const recalibrateButton = document.getElementById('recalibrateButton');
const statusMessage = document.getElementById('statusMessage');

// Create a DrawingUtils object to draw landmarks
const drawingUtils = new DrawingUtils(canvasCtx);

let poseLandmarker = undefined;
let runningMode = "VIDEO";

// --- New State Variables ---
let isVisualizationEnabled = true; // State for visualization toggle
let isTrackingEnabled = false; // State for main tracking toggle
let isCalibrating = false; // State for calibration process
let calibrationData = []; // Store data points during calibration
let baselineVerticalDistance = null; // The averaged baseline for good posture

const CALIBRATION_DURATION_SECONDS = 3;
const CHECK_INTERVAL_SECONDS = 3;
const VERTICAL_DECREASE_PERCENTAGE_THRESHOLD = 0.10; // 10% decrease

// Global interval IDs to manage our timed processes
let mainPredictionIntervalId = null;
let calibrationIntervalId = null;

// Function to calculate simple vertical distance (y-coordinate difference)
function calculateVerticalDistance(point1, point2) {
    if (!point1 || !point2) return 0;
    // We care about the absolute difference in y-coordinates
    return Math.abs(point1.y - point2.y);
}

// Function to draw video and landmarks
function drawCanvas(landmarks, color) {
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

    // Draw the video frame
    if (isVisualizationEnabled) {
        canvasCtx.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);
    }

    // Draw landmarks if they exist and we are visualizing
    if (landmarks && isVisualizationEnabled) {
        drawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, {
            color: color,
            lineWidth: 4
        });
        drawingUtils.drawLandmarks(landmarks, {
            color: color,
            lineWidth: 2
        });
    }
    canvasCtx.restore();
}

// Initialize the PoseLandmarker model
async function createPoseLandmarker() {
    statusMessage.textContent = "Loading AI model...";
    try {
        const vision = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
        );
        poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
                delegate: "GPU"
            },
            runningMode: runningMode,
            outputSegmentationMasks: false
        });
        statusMessage.textContent = "Model loaded. Click 'Start Camera'.";
        startCameraButton.disabled = false;
    } catch (err) {
        statusMessage.textContent = `Error loading model: ${err.message}`;
        console.error("Error loading PoseLandmarker model:", err);
    }
}

// Function to start the webcam
async function enableCam() {
    if (!poseLandmarker) {
        statusMessage.textContent = "Model not loaded yet. Please wait.";
        return;
    }

    statusMessage.textContent = "Requesting camera access...";
    const constraints = { video: true };

    try {
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        videoElement.srcObject = stream;
        videoElement.addEventListener("loadeddata", () => {
            canvasElement.width = videoElement.videoWidth;
            canvasElement.height = videoElement.videoHeight;
            // The initial state is 'paused' until calibration happens
            toggleTrackingButton.textContent = 'Resume warning';
            isTrackingEnabled = false;
            // Now start the calibration process immediately after the camera is ready
            startCalibration();
        });
        startCameraButton.textContent = "App is running";
        startCameraButton.disabled = true;
    } catch (err) {
        statusMessage.textContent = `Error accessing camera: ${err.name} - ${err.message}. Please allow camera access.`;
        console.error("Error accessing camera:", err);
    }
}

// Start the calibration process
function startCalibration() {
    isCalibrating = true;
    calibrationData = [];
    statusMessage.textContent = `Calibrating good posture... Stand straight for ${CALIBRATION_DURATION_SECONDS} seconds.`;
    toggleTrackingButton.disabled = true;
    recalibrateButton.disabled = true;

    let countdown = CALIBRATION_DURATION_SECONDS;
    calibrationIntervalId = setInterval(() => {
        if (countdown > 0) {
            statusMessage.textContent = `Calibrating good posture... ${countdown}s left.`;
            // Get current pose data for calibration
            const result = poseLandmarker.detectForVideo(videoElement, performance.now());
            if (result.landmarks && result.landmarks.length > 0) {
                const landmark = result.landmarks[0];
                const leftEar = landmark[7];
                const leftShoulder = landmark[11];
                const rightEar = landmark[8];
                const rightShoulder = landmark[12];
                if (leftEar && leftShoulder && rightEar && rightShoulder) {
                    const avgDist = (calculateVerticalDistance(leftEar, leftShoulder) + calculateVerticalDistance(rightEar, rightShoulder)) / 2;
                    calibrationData.push(avgDist);
                }
            }
            drawCanvas(result.landmarks[0], '#00FFFF'); // Cyan for calibration
            countdown--;
        } else {
            // Calibration complete
            clearInterval(calibrationIntervalId);
            isCalibrating = false;
            if (calibrationData.length > 0) {
                const sum = calibrationData.reduce((acc, dist) => acc + dist, 0);
                baselineVerticalDistance = sum / calibrationData.length;
                statusMessage.textContent = `Calibration complete! Baseline established.`;
                isTrackingEnabled = true;
                toggleTrackingButton.textContent = 'Warning active';
                // Start the main posture check loop
                startPostureCheck();
            } else {
                statusMessage.textContent = "Calibration failed: No person detected. Please try again.";
                baselineVerticalDistance = null;
            }
            toggleTrackingButton.disabled = false;
            recalibrateButton.disabled = false;
        }
    }, 1000); // Run every second
}

// Main posture check function
function startPostureCheck() {
    // Clear any existing interval to prevent duplicates
    if (mainPredictionIntervalId) {
        clearInterval(mainPredictionIntervalId);
    }

    mainPredictionIntervalId = setInterval(() => {
        if (!isTrackingEnabled || isCalibrating || !baselineVerticalDistance) {
            // If tracking is paused or we're calibrating, just return and do nothing
            return;
        }

        const currentTime = performance.now();
        const result = poseLandmarker.detectForVideo(videoElement, currentTime);
        let currentDisplayColor = '#00FF00'; // Default good posture color

        if (result.landmarks && result.landmarks.length > 0) {
            const landmark = result.landmarks[0];
            const leftEar = landmark[7];
            const leftShoulder = landmark[11];
            const rightEar = landmark[8];
            const rightShoulder = landmark[12];

            if (leftEar && leftShoulder && rightEar && rightShoulder) {
                const currentAvgDist = (calculateVerticalDistance(leftEar, leftShoulder) + calculateVerticalDistance(rightEar, rightShoulder)) / 2;
                
                // Check for slouching
                const decreasePercentage = (baselineVerticalDistance - currentAvgDist) / baselineVerticalDistance;
                if (decreasePercentage > VERTICAL_DECREASE_PERCENTAGE_THRESHOLD) {
                    statusMessage.textContent = "Slouching! Sit straight!";
                    currentDisplayColor = '#FF0000'; // Red for confirmed slouch
                    playAlertSound();
                } else {
                    statusMessage.textContent = "Good posture!";
                }
            } else {
                statusMessage.textContent = "Adjust to see your ears and shoulders!";
                currentDisplayColor = '#FFFF00'; // Yellow if landmarks are missing
            }
            // Draw the current state
            drawCanvas(landmark, currentDisplayColor);
        } else {
            // No person detected
            statusMessage.textContent = "No person detected. Please adjust your position.";
            currentDisplayColor = '#808080'; // Grey if no person
            drawCanvas(null, currentDisplayColor); // Still need to draw the video frame
        }

    }, CHECK_INTERVAL_SECONDS * 1000); // Run every 3 seconds
}

// Function to play an alert sound
const alertSound = new Audio('bark.wav');
let soundPlaying = false;
function playAlertSound() {
    if (!soundPlaying) {
        soundPlaying = true;
        alertSound.play().catch(e => console.error("Error playing sound:", e));
        // Reset soundPlaying after a short cooldown
        setTimeout(() => { soundPlaying = false; }, 3000);
    }
}

// --- Event Listeners for new buttons ---
startCameraButton.addEventListener('click', enableCam);

toggleVisualizationButton.addEventListener('click', () => {
    isVisualizationEnabled = !isVisualizationEnabled;
    toggleVisualizationButton.textContent = isVisualizationEnabled ? 'Coordinate visualization off' : 'Coordinate visualization on';
    // When turning off visualization, clear the canvas
    if (!isVisualizationEnabled) {
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    } else {
        // Force a redraw to show the current state
        startPostureCheck();
    }
});

toggleTrackingButton.addEventListener('click', () => {
    isTrackingEnabled = !isTrackingEnabled;
    toggleTrackingButton.textContent = isTrackingEnabled ? 'Warning active' : 'Warning inactive';
    if (!isTrackingEnabled) {
        statusMessage.textContent = "Tracking paused.";
        // Clear the interval if tracking is paused
        if (mainPredictionIntervalId) {
            clearInterval(mainPredictionIntervalId);
        }
    } else {
        statusMessage.textContent = "Tracking resumed.";
        // Restart the interval if tracking is resumed
        startPostureCheck();
    }
});

recalibrateButton.addEventListener('click', () => {
    // Stop any existing tracking loop and start the calibration process again
    if (mainPredictionIntervalId) {
        clearInterval(mainPredictionIntervalId);
    }
    startCalibration();
});

// Initial call to load the model
createPoseLandmarker();