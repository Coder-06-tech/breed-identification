// static/js/camera.js

let webcamStream = null;

export function initCamera(onPhotoSnapped) {
    const videoEl = document.getElementById('webcam');
    const webcamContainer = document.getElementById('webcamContainer');
    const webcamPlaceholder = document.getElementById('webcamPlaceholder');
    const btnStartCamera = document.getElementById('btnStartCamera');
    const btnCapturePhoto = document.getElementById('btnCapturePhoto');
    const btnStopCamera = document.getElementById('btnStopCamera');

    if (!btnStartCamera) return;

    const stopStream = () => {
        if (webcamStream) {
            webcamStream.getTracks().forEach(track => track.stop());
            webcamStream = null;
        }
        if (videoEl) {
            videoEl.srcObject = null;
            videoEl.style.display = 'none';
        }
        if (webcamPlaceholder) {
            webcamPlaceholder.style.display = 'flex';
        }
        btnStartCamera.style.display = 'inline-flex';
        btnCapturePhoto.style.display = 'none';
        btnStopCamera.style.display = 'none';
        if (webcamContainer) {
            webcamContainer.classList.remove('scanning');
        }
    };

    btnStartCamera.addEventListener('click', async () => {
        try {
            webcamStream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: 'environment' },
                audio: false
            });
            if (videoEl) {
                videoEl.srcObject = webcamStream;
                videoEl.style.display = 'block';
            }
            if (webcamPlaceholder) {
                webcamPlaceholder.style.display = 'none';
            }
            btnStartCamera.style.display = 'none';
            btnCapturePhoto.style.display = 'inline-flex';
            btnStopCamera.style.display = 'inline-flex';
        } catch (err) {
            alert("Camera Access Failed: Make sure permissions are allowed. Details: " + err.message);
        }
    });

    btnStopCamera.addEventListener('click', stopStream);

    btnCapturePhoto.addEventListener('click', () => {
        if (!webcamStream || !videoEl) return;
        
        if (webcamContainer) {
            webcamContainer.classList.add('scanning');
        }
        
        const canvas = document.createElement('canvas');
        canvas.width = videoEl.videoWidth || 640;
        canvas.height = videoEl.videoHeight || 480;
        
        const ctx = canvas.getContext('2d');
        ctx.translate(canvas.width, 0);
        ctx.scale(-1, 1);
        ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height);
        
        stopStream();
        
        canvas.toBlob((blob) => {
            if (!blob) {
                alert("Failed to capture image frame.");
                return;
            }
            onPhotoSnapped(blob);
        }, 'image/jpeg', 0.95);
    });

    // Handle navigation events to shut down camera if active
    document.addEventListener('spa-navigate', () => {
        stopStream();
    });
}

export function shutdownCamera() {
    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
        webcamStream = null;
    }
}
