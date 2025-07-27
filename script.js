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

// Initialize the PoseLandmarker model
async function createPoseLandmarker() {
    statusMessage.textContent = "Loading AI model...";
    const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
    );
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: "app/shared/models/pose_landmarker_full.task", // Path to the model file
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
    // Get user media (webcam access)
    const constraints = { video: true }; // Request video only

    try {
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        videoElement.srcObject = stream;
        videoElement.addEventListener("loadeddata", predictWebcam); // Start prediction once video data loads
        statusMessage.textContent = "Camera started. Adjust your position.";
        startCameraButton.textContent = "Camera On";
        startCameraButton.disabled = true; // Disable button once camera starts
    } catch (err) {
        statusMessage.textContent = `Error accessing camera: ${err.name} - ${err.message}. Please allow camera access.`;
        console.error("Error accessing camera:", err);
    }
}

// Prediction loop
async function predictWebcam() {
    // Set canvas dimensions to match video
    canvasElement.style.width = videoElement.videoWidth + "px";
    canvasElement.style.height = videoElement.videoHeight + "px";
    canvasElement.width = videoElement.videoWidth;
    canvasElement.height = videoElement.videoHeight;

    let startTimeMs = performance.now(); // Get current timestamp for MediaPipe

    // Only process new video frames
    if (lastVideoTime !== videoElement.currentTime) {
        lastVideoTime = videoElement.currentTime;
        poseLandmarker.detectForVideo(videoElement, startTimeMs, (result) => {
            canvasCtx.save();
            canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height); // Clear previous drawings

            // Draw the video frame on the canvas (important for mirroring effect)
            // Note: drawingUtils.drawConnectors and drawLandmarks will automatically handle mirroring
            // if the video element itself is mirrored with transform: scaleX(-1);
            canvasCtx.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);

            if (result.landmarks && result.landmarks.length > 0) {
                for (const landmark of result.landmarks) {
                    // Draw the skeleton and landmarks
                    drawingUtils.drawConnectors(landmark, PoseLandmarker.POSE_CONNECTIONS, { color: '#00FF00', lineWidth: 4 });
                    drawingUtils.drawLandmarks(landmark, { color: '#FF0000', lineWidth: 2 });

                    // --- Posture Analysis Logic (to be expanded) ---
                    // Example: Check for forward head posture
                    const leftShoulder = landmark[11]; // MediaPipe landmark indices
                    const rightShoulder = landmark[12];
                    const leftEar = landmark[7];
                    const rightEar = landmark[8];

                    // Simple check: Is the ear significantly in front of the shoulder?
                    // This is very basic and will need refinement!
                    // Coordinates are normalized 0-1, so differences are small.
                    const isSlouching = (leftEar.x > leftShoulder.x + 0.03) || (rightEar.x < rightShoulder.x - 0.03); // Assuming mirrored view

                    if (isSlouching) {
                        statusMessage.textContent = "Slouching! Sit straight!";
                        // You'd add your sound playing logic here
                        // For now, let's change the skeleton color to red
                        drawingUtils.drawConnectors(landmark, PoseLandmarker.POSE_CONNECTIONS, { color: '#FF0000', lineWidth: 4 });
                        drawingUtils.drawLandmarks(landmark, { color: '#FF0000', lineWidth: 2 });
                        playAlertSound(); // Function to be defined
                    } else {
                        statusMessage.textContent = "Good posture!";
                    }
                }
            } else {
                statusMessage.textContent = "No person detected. Please adjust your position.";
            }
            canvasCtx.restore();
        });
    }

    // Call this function again to continuously process frames
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