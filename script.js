// Reference: https://codepen.io/mediapipe-preview/pen/abRLMxN
import { PoseLandmarker, FilesetResolver, DrawingUtils } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";

const videoElement = document.getElementById('webcamVideo');
const canvasElement = document.getElementById('outputCanvas');
const canvasCtx = canvasElement.getContext('2d');
const startCameraButton = document.getElementById('startCameraButton');
const statusMessage = document.getElementById('statusMessage');

// Create a DrawingUtils object to draw landmarks
const drawingUtils = new DrawingUtils(canvasCtx);

let poseLandmarker = undefined; // Will hold our PoseLandmarker instance
let runningMode = "VIDEO"; // MediaPipe running mode for live video
let lastVideoTime = -1; // To ensure we only process new video frames

// --- New Posture Detection Variables ---
// Store timestamped vertical distances for dynamic comparison
const VERTICAL_DISTANCE_HISTORY_DURATION_MS = 2500; // Keep history for 2.5 seconds
const INITIAL_CALIBRATION_DURATION_MS = 5000; // Assume good posture for the first 5 seconds
let poseDataHistory = []; // Stores { timestamp: ms, leftEarY: y, leftShoulderY: y, rightEarY: y, rightShoulderY: y }
let startTime = null; // To track calibration duration

const VERTICAL_DECREASE_PERCENTAGE_THRESHOLD = 0.10; // 10% decrease
const SLOUCH_PERSISTENCE_FRAMES = 10; // How many consecutive frames of bad posture to trigger alert (remains for stability)
let consecutiveSlouchFrames = 0;

// Function to calculate simple vertical distance (y-coordinate difference)
function calculateVerticalDistance(point1, point2) {
    if (!point1 || !point2) return 0;
    // We care about the absolute difference in y-coordinates
    return Math.abs(point1.y - point2.y);
}

// Initialize the PoseLandmarker model
async function createPoseLandmarker() {
    statusMessage.textContent = "Loading AI model...";
    const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
    );
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
            // Using the full model by default.
            // Switch to 'pose_landmarker_lite.task' for better performance on less powerful devices (e.g., phones).
            // Use API for ease of use, or download the model file and provide the path.
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
            // modelAssetPath: "app/models/pose_landmarker_full.task", // Local path to the model
            delegate: "GPU" // Try "GPU" for better performance, fallbacks to "CPU"
        },
        runningMode: runningMode,
        outputSegmentationMasks: false // Optional: if you want a mask of the person
    });
    statusMessage.textContent = "Model loaded. Click 'Start Camera'.";
    startCameraButton.disabled = false; // Enable button once model is loaded
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
            startTime = performance.now(); // Record start time for calibration
            predictWebcam(); // Start prediction once video data loads
        });
        statusMessage.textContent = "Camera started. Adjust your position.";
        startCameraButton.textContent = "Camera On";
        startCameraButton.disabled = true; // Disable button once camera starts
        // Reset state for new session
        poseDataHistory = [];
        consecutiveSlouchFrames = 0;
    } catch (err) {
        statusMessage.textContent = `Error accessing camera: ${err.name} - ${err.message}. Please allow camera access.`;
        console.error("Error accessing camera:", err);
    }
}

// Prediction loop
async function predictWebcam() {
    canvasElement.style.width = videoElement.videoWidth + "px";
    canvasElement.style.height = videoElement.videoHeight + "px";
    canvasElement.width = videoElement.videoWidth;
    canvasElement.height = videoElement.videoHeight;

    let currentTime = performance.now(); // Get current timestamp for MediaPipe and history tracking

    if (lastVideoTime !== videoElement.currentTime) {
        lastVideoTime = videoElement.currentTime;
        poseLandmarker.detectForVideo(videoElement, currentTime, (result) => {
            canvasCtx.save();
            canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
            canvasCtx.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);

            let isCurrentlySlouching = false;
            let currentDisplayColor = '#00FF00'; // Default good posture color

            if (result.landmarks && result.landmarks.length > 0) {
                const landmark = result.landmarks[0]; // Assuming one person

                // Store current landmark data for history
                // indices: leftEar: 7, leftShoulder: 11, rightEar: 8, rightShoulder: 12
                const leftEar = landmark[7];
                const leftShoulder = landmark[11];
                const rightEar = landmark[8];
                const rightShoulder = landmark[12];

                if (leftEar && leftShoulder && rightEar && rightShoulder) {
                    const currentLeftVerticalDist = calculateVerticalDistance(leftEar, leftShoulder);
                    const currentRightVerticalDist = calculateVerticalDistance(rightEar, rightShoulder);
                    const currentAverageVerticalDist = (currentLeftVerticalDist + currentRightVerticalDist) / 2;

                    // Add current pose data to history
                    poseDataHistory.push({
                        timestamp: currentTime,
                        verticalDist: currentAverageVerticalDist
                    });

                    // Prune old data from history
                    poseDataHistory = poseDataHistory.filter(data => currentTime - data.timestamp < VERTICAL_DISTANCE_HISTORY_DURATION_MS);

                    // --- Posture Analysis Logic ---
                    if (currentTime - startTime < INITIAL_CALIBRATION_DURATION_MS) {
                        // Calibration period
                        statusMessage.textContent = `Calibrating good posture... (${Math.ceil((INITIAL_CALIBRATION_DURATION_MS - (currentTime - startTime)) / 1000)}s left)`;
                        currentDisplayColor = '#00FFFF'; // Cyan for calibration
                        consecutiveSlouchFrames = 0; // Ensure no alerts during calibration
                    } else if (poseDataHistory.length > 0) {
                        // Get average vertical distance from the recent past (e.g., 2 seconds ago)
                        const referenceTime = currentTime - VERTICAL_DISTANCE_HISTORY_DURATION_MS;
                        const relevantPastData = poseDataHistory.filter(data => data.timestamp < referenceTime);

                        if (relevantPastData.length > 0) {
                            const sumPastDist = relevantPastData.reduce((acc, data) => acc + data.verticalDist, 0);
                            const averagePastDist = sumPastDist / relevantPastData.length;

                            if (averagePastDist > 0) { // Avoid division by zero
                                const decreasePercentage = (averagePastDist - currentAverageVerticalDist) / averagePastDist;

                                if (decreasePercentage > VERTICAL_DECREASE_PERCENTAGE_THRESHOLD) {
                                    isCurrentlySlouching = true;
                                    // console.log(`Slouch detected! Decrease: ${(decreasePercentage * 100).toFixed(2)}%`);
                                }
                            }
                        } else {
                            // Not enough history to compare yet, assume good posture
                            statusMessage.textContent = "Building history...";
                        }
                    }

                    if (isCurrentlySlouching) {
                        consecutiveSlouchFrames++;
                        if (consecutiveSlouchFrames >= SLOUCH_PERSISTENCE_FRAMES) {
                            statusMessage.textContent = "Slouching! Sit straight!";
                            currentDisplayColor = '#FF0000'; // Red for confirmed slouch
                            playAlertSound();
                        } else {
                            // Still show orange if slouching, but not enough to alert yet
                            currentDisplayColor = '#FFA500'; // Orange for detected but not alerted
                            statusMessage.textContent = `Slouch detected... (${SLOUCH_PERSISTENCE_FRAMES - consecutiveSlouchFrames} frames left)`;
                        }
                    } else {
                        statusMessage.textContent = "Good posture!";
                        consecutiveSlouchFrames = 0; // Reset counter if posture is good
                        currentDisplayColor = '#00FF00'; // Green for good posture
                    }
                } else {
                    // Not all required landmarks detected
                    statusMessage.textContent = "Adjust to see your ears and shoulders!";
                    consecutiveSlouchFrames = 0; // Reset
                    currentDisplayColor = '#FFFF00'; // Yellow if landmarks are missing
                }

                // Draw the skeleton and landmarks with updated color
                drawingUtils.drawConnectors(landmark, PoseLandmarker.POSE_CONNECTIONS, {
                    color: currentDisplayColor,
                    lineWidth: 4
                });
                drawingUtils.drawLandmarks(landmark, {
                    color: currentDisplayColor,
                    lineWidth: 2
                });

            } else {
                statusMessage.textContent = "No person detected. Please adjust your position.";
                consecutiveSlouchFrames = 0; // Reset if no person detected
                poseDataHistory = []; // Clear history
                startTime = currentTime; // Reset start time if person disappears
                currentDisplayColor = '#808080'; // Grey if no person
            }
            canvasCtx.restore();
        });
    }

    window.requestAnimationFrame(predictWebcam);
}

// Function to play an alert sound
const alertSound = new Audio('bark.wav'); // Make sure the file is in your project root
let soundPlaying = false;
function playAlertSound() {
    if (!soundPlaying) {
        soundPlaying = true;
        alertSound.play().catch(e => console.error("Error playing sound:", e));
        setTimeout(() => { soundPlaying = false; }, 3000); // Cooldown for 3 seconds
    }
}

// Event listener for the start camera button
startCameraButton.addEventListener('click', enableCam);

// Initial call to load the model
createPoseLandmarker();