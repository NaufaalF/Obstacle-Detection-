// static/script.js
const socket = io();
const video = document.getElementById('video');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');

const latencyText = document.getElementById('latency')
const avgLatencyText = document.getElementById('avg-latency')

const latencyPlot = document.getElementById('latencyPlot');

socket.on('latency_plot', data => {
    latencyPlot.src = "data:image/png;base64," + data.img;
});

let latencySum = 0;
let latencyCount = 0;

let streaming = false;
let fps = parseInt(document.getElementById('fps').value);

// Ambil elemen audio dari index.html
const alertSound = document.getElementById('alertSound');

const ctxChart = document.getElementById('latencyChart').getContext('2d');
const latencyData = {
    labels: [],
    datasets: [{
        label: 'Latency (ms)',
        data: [],
        borderColor: 'lime',
        borderWidth: 2,
        fill: false
    }]
};

const latencyChart = new Chart(ctxChart, {
    type: 'line',
    data: latencyData,
    options: {
        animation: false,
        scales: {
            x: { title: { display: true, text: 'Time (s)' } },
            y: { title: { display: true, text: 'Latency (ms)' } }
        }
    }
});

// Update FPS when selection changes
document.getElementById('fps').addEventListener('change', e => {
    fps = parseInt(e.target.value);
});

// // Start camera
// const startBtn = document.getElementById('startBtn');
// startBtn.addEventListener('click', async () => {
//     if (navigator.mediaDevices.getUserMedia) {
//         const stream = await navigator.mediaDevices.getUserMedia({ video: true });
//         video.srcObject = stream;
//         streaming = true;
//         sendFrames();
//     }
// });

// Start camera
const startBtn = document.getElementById('startBtn');
startBtn.addEventListener('click', async () => {
    if (navigator.mediaDevices.getUserMedia) {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: { exact: "environment" } } // back camera
            });
            video.srcObject = stream;
            streaming = true;
            sendFrames();
        } catch (err) {
            console.error("Back camera not available, using default camera.", err);
            // fallback to default if back camera not found
            const fallbackStream = await navigator.mediaDevices.getUserMedia({ video: true });
            video.srcObject = fallbackStream;
            streaming = true;
            sendFrames();
        }
    }
});

// Stop camera
const stopBtn = document.getElementById('stopBtn');
stopBtn.addEventListener('click', () => {
    const stream = video.srcObject;
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }
    streaming = false;
    socket.emit('plot');
});

// Send frames to server
function sendFrames() {
    if (!streaming) return;
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = video.videoWidth;
    tempCanvas.height = video.videoHeight;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(video, 0, 0);
    const dataURL = tempCanvas.toDataURL('image/jpeg');

    const timestamp = Date.now(); // mark time when frame is sent
    socket.emit('frame', { image: dataURL, ts: timestamp });
    setTimeout(sendFrames, 1000 / fps);
}

// Receive detections from server
socket.on('detections', data => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.lineWidth = 2;
    ctx.font = '16px Arial';
    ctx.strokeStyle = 'lime';
    ctx.fillStyle = 'lime';

    let playAlert = false; // flag untuk mainkan suara

    data.detections.forEach(det => {
        const [x1, y1, x2, y2] = det.bbox;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        const label = `${det.class_name} ${(det.distance_cm ? det.distance_cm.toFixed(0) + 'cm' : '')}`;
        ctx.fillText(label, x1, y1 > 20 ? y1 - 5 : y1 + 15);

        // Cek alert dari server
        if (det.alert) {
            playAlert = true;
        }
    });

    // Latency calculation
    if (data.ts) {
        const latency = Date.now() - data.ts;
        latencyText.innerText = `Latency: ${latency} ms`;

        // Update average latency
        latencySum += latency;
        latencyCount++;
        const avgLatency = (latencySum / latencyCount).toFixed(1);
        avgLatencyText.innerText = `Avg Latency: ${avgLatency} ms`;

        // plot current latency
        updateLatencyChart(latency);

        socket.emit('latency', { latency: latency }); // send to server
    }

    // Mainkan suara jika ada alert
    if (playAlert && alertSound) {
        alertSound.currentTime = 0; // reset audio
        alertSound.play().catch(err => console.log(err));
    }

    // Update chart with new latency
    function updateLatencyChart(latency) {
        const now = new Date();
        latencyData.labels.push(now.toLocaleTimeString());
        latencyData.datasets[0].data.push(latency);

        // keep last 60 seconds (assuming ~1 fps latency update)
        if (latencyData.labels.length > 60) {
            latencyData.labels.shift();
            latencyData.datasets[0].data.shift();
        }

        latencyChart.update();
    }
});
