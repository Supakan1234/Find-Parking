import cv2
import numpy as np
import json
import os
import time
import threading
import csv
import copy
from datetime import datetime
from flask import Flask, Response, jsonify, request

# --- [ตั้งค่าพื้นฐาน] ---
app = Flask(__name__)
SAVE_FILE = "parking_master_data.json"
LOG_FILE = "parking_history.csv"
BUSY_RATIO_THRESHOLD = 0.12 
RESERVE_TIMEOUT = 90 
OCCUPY_DELAY = 3 

# --- [ตัวแปรควบคุมระบบ] ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

parking_stats = {"free": 0, "total": 0, "reserved": 0, "occupied": 0}
slots = [] 
temp_points = [] 
encoded_frame = None 
lock = threading.Lock()

# Editor States
selected_slot = -1
selected_point = -1 
is_dragging = False
last_mouse_pos = (0, 0)
mouse_curr = (0, 0)
move_all_mode = False
last_status = []
copied_slot_data = None 

# --- [ระบบจัดการข้อมูล & Logging] ---
def load_data():
    if os.path.exists(SAVE_FILE):
        try:
            with open(SAVE_FILE, "r") as f: return json.load(f)
        except: return []
    return []

def save_data(data):
    clean_data = [[s[0], 0, 0, 0, False, s[5] if len(s)>5 else "18.573077155796554,99.00090305034648", s[6] if len(s)>6 else False] for s in data]
    with open(SAVE_FILE, "w") as f: json.dump(clean_data, f)

def log_event(slot_id, event_type, status_code):
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Slot_ID", "Event", "Status_Code"])
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([now, slot_id + 1, event_type, status_code])

slots = load_data()

# --- [Mouse Logic] ---
def mouse_events(event, x, y, flags, param):
    global slots, temp_points, selected_slot, selected_point, is_dragging, last_mouse_pos, mouse_curr
    mouse_curr = (x, y)
    
    if event == cv2.EVENT_RBUTTONDOWN: 
        temp_points.append([x, y])
        if len(temp_points) == 4:
            slots.append([temp_points, 0, 0, 0, False, "18.573077155796554,99.00090305034648", False])
            temp_points = []; save_data(slots)
            
    elif event == cv2.EVENT_LBUTTONDOWN:
        is_dragging, last_mouse_pos = True, (x, y)
        selected_slot, selected_point = -1, -1
        for i, s in enumerate(slots):
            for j, pt in enumerate(s[0]):
                if np.linalg.norm(np.array(pt) - np.array([x, y])) < 12:
                    selected_slot, selected_point = i, j; return
        for i, s in enumerate(slots):
            if cv2.pointPolygonTest(np.array(s[0], np.int32), (x, y), False) >= 0:
                selected_slot = i; break
                
    elif event == cv2.EVENT_MOUSEMOVE and is_dragging:
        dx, dy = x - last_mouse_pos[0], y - last_mouse_pos[1]
        if move_all_mode:
            for s in slots:
                for pt in s[0]: pt[0] += dx; pt[1] += dy
        elif selected_slot != -1:
            target_slot = slots[selected_slot]
            is_sym = target_slot[6] if len(target_slot) > 6 else False
            if selected_point != -1:
                if is_sym:
                    pts = np.array(target_slot[0])
                    center = np.mean(pts, axis=0)
                    dist_x = pts[selected_point][0] - center[0]
                    dist_y = pts[selected_point][1] - center[1]
                    scale_x = 1 + (dx / dist_x) if abs(dist_x) > 1 else 1
                    scale_y = 1 + (dy / dist_y) if abs(dist_y) > 1 else 1
                    for i in range(4):
                        pts[i][0] = int(center[0] + (pts[i][0] - center[0]) * scale_x)
                        pts[i][1] = int(center[1] + (pts[i][1] - center[1]) * scale_y)
                    slots[selected_slot][0] = pts.tolist()
                else:
                    slots[selected_slot][0][selected_point][0] += dx
                    slots[selected_slot][0][selected_point][1] += dy
            else: 
                for pt in target_slot[0]: pt[0] += dx; pt[1] += dy
        last_mouse_pos = (x, y)
        
    elif event == cv2.EVENT_LBUTTONUP:
        if is_dragging: save_data(slots)
        is_dragging = False

# --- [วิเคราะห์วิดีโอ & คีย์บอร์ด] ---
def main_process():
    global slots, encoded_frame, parking_stats, selected_slot, move_all_mode, last_status, copied_slot_data
    cv2.namedWindow("Setup")
    cv2.setMouseCallback("Setup", mouse_events)
    last_analysis = 0

    while True:
        success, img = cap.read()
        if not success: continue
        img_display, curr = img.copy(), time.time()

        if curr - last_analysis > 0.5:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_thr = cv2.adaptiveThreshold(cv2.GaussianBlur(img_gray,(3,3),1), 255, 0, 1, 25, 10)
            f_count, r_count, o_count = 0, 0, 0
            
            with lock:
                if not last_status: last_status = [s[1] for s in slots]
                while len(last_status) < len(slots): last_status.append(0)

                for i, s in enumerate(slots):
                    pts = np.array(s[0], np.int32)
                    mask = np.zeros(img_thr.shape, dtype=np.uint8)
                    cv2.fillPoly(mask, [pts], 255)
                    area = cv2.contourArea(pts)
                    busy = (cv2.countNonZero(cv2.bitwise_and(img_thr, mask))/area > BUSY_RATIO_THRESHOLD) if area>0 else False
                    if busy:
                        if s[3] == 0: slots[i][3] = curr
                        dwell = curr - s[3]
                    else: 
                        slots[i][3] = 0
                        dwell = 0
                    
                    if s[1] == 2: 
                        r_count += 1
                        slots[i][4] = True if dwell > OCCUPY_DELAY else False
                        if (curr - s[2]) > RESERVE_TIMEOUT: 
                            slots[i][1], slots[i][2], slots[i][3], slots[i][4] = 0, 0, 0, False
                    else:
                        if dwell > OCCUPY_DELAY: 
                            slots[i][1] = 1; o_count += 1
                        else: 
                            slots[i][1] = 0; f_count += 1
                    
                    if i < len(last_status) and slots[i][1] != last_status[i]:
                        labels = {0: "Slot Freed", 1: "Car Occupied", 2: "Reserved"}
                        log_event(i, labels.get(slots[i][1], "Unknown"), slots[i][1])
                        last_status[i] = slots[i][1]

                parking_stats.update({"free": f_count, "total": len(slots), "reserved": r_count, "occupied": o_count})
            last_analysis = curr

        for i, s in enumerate(slots):
            is_sel = (i == selected_slot or move_all_mode)
            color = (255, 255, 255) if is_sel else ((0,255,0) if s[1]==0 else (0,255,255) if s[1]==2 else (0,0,255))
            cv2.polylines(img_display, [np.array(s[0], np.int32)], True, color, 2)
            cv2.putText(img_display, str(i+1), tuple(np.mean(s[0], axis=0).astype(int)), 1, 1, color, 2)
            if i == selected_slot:
                for pt in s[0]: cv2.circle(img_display, tuple(pt), 4, (255, 255, 255), -1)
        
        _, buf = cv2.imencode('.jpg', img_display, [cv2.IMWRITE_JPEG_QUALITY, 80]) 
        with lock: encoded_frame = buf.tobytes()
        cv2.imshow("Setup", img_display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('m'): move_all_mode = not move_all_mode
        elif key == ord('z'):
            mx, my = mouse_curr
            slots.append([[ [mx-40, my-25], [mx+40, my-25], [mx+40, my+25], [mx-40, my+25] ], 0, 0, 0, False, "18.573077155796554,99.00090305034648", True])
            last_status.append(0); save_data(slots)
        elif key == ord('x'):
            if selected_slot != -1:
                slots.pop(selected_slot); last_status.pop(selected_slot)
                selected_slot = -1; save_data(slots)
        elif key == ord('c'):
            if selected_slot != -1: copied_slot_data = copy.deepcopy(slots[selected_slot])
        elif key == ord('v'):
            if copied_slot_data:
                mx, my = mouse_curr
                pts = np.array(copied_slot_data[0])
                offset = np.array([mx, my]) - np.mean(pts, axis=0)
                new_pts = (pts + offset).astype(int).tolist()
                is_sym = copied_slot_data[6] if len(copied_slot_data) > 6 else False
                slots.append([new_pts, 0, 0, 0, False, "18.573077155796554,99.00090305034648", is_sym])
                last_status.append(0); save_data(slots)
        elif key == ord('k'):
            slots = []; last_status = []; selected_slot = -1; save_data(slots)

# --- [API Routes] ---
@app.route('/api/hourly_stats')
def get_hourly_stats():
    today_stats = [0] * 24
    history_data = {} 
    today_str = datetime.now().strftime("%Y-%m-%d")
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) < 3: continue
                try:
                    dt = datetime.strptime(row[0].strip(), "%Y-%m-%d %H:%M:%S")
                    d_key = dt.strftime("%Y-%m-%d")
                    if row[2].strip() == "Car Occupied":
                        if d_key == today_str: today_stats[dt.hour] += 1
                        if d_key not in history_data: history_data[d_key] = [0] * 24
                        history_data[d_key][dt.hour] += 1
                except: continue
    avg_stats = [round(sum(d[h] for d in history_data.values())/len(history_data),1) if history_data else 0 for h in range(24)]
    return jsonify({"today": today_stats, "average": avg_stats})

@app.route('/api/all_data')
def get_all_data():
    curr = time.time()
    with lock:
        res = [{"id": i, "status": s[1], "remaining": max(0, int(RESERVE_TIMEOUT - (curr - s[2]))) if s[1] == 2 else 0, "is_arrived": s[4], "gps": s[5]} for i, s in enumerate(slots)]
    return jsonify({"stats": parking_stats, "slots": res})

@app.route('/api/reserve', methods=['POST'])
def reserve_slot():
    sid = int(request.json.get('slot_id'))
    with lock:
        if 0 <= sid < len(slots) and slots[sid][1] == 0:
            slots[sid][1], slots[sid][2], slots[sid][3], slots[sid][4] = 2, time.time(), 0, False
            return jsonify({"status": "success"})
    return jsonify({"status": "error"}), 400

@app.route('/api/extend', methods=['POST'])
def extend_slot():
    sid = int(request.json.get('slot_id'))
    with lock:
        if 0 <= sid < len(slots) and slots[sid][1] == 2:
            slots[sid][2] = time.time() 
            return jsonify({"status": "success"})
    return jsonify({"status": "error"}), 400

@app.route('/api/cancel', methods=['POST'])
def cancel_slot():
    sid = int(request.json.get('slot_id'))
    with lock:
        if 0 <= sid < len(slots):
            slots[sid][1], slots[sid][2], slots[sid][3], slots[sid][4] = 0, 0, 0, False
            return jsonify({"status": "success"})
    return jsonify({"status": "error"}), 400

@app.route('/video_feed')
def video_feed():
    def gen():
        while True:
            time.sleep(0.05)
            with lock:
                if encoded_frame: yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + encoded_frame + b'\r\n')
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- [ส่วนของ index() ที่แก้ไขใหม่] ---

@app.route('/')
def index():
    return """
    <html>
    <head>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Kanit:wght@300;400;600&display=swap');
            :root { --primary: linear-gradient(135deg, #6366f1 0%, #a855f7 100%); --success: #10b981; --warning: #f59e0b; --danger: #ef4444; --bg: #0f172a; --card: rgba(30, 41, 59, 0.7); --glass-border: rgba(255, 255, 255, 0.1); }
            body { font-family: 'Kanit', sans-serif; background: #0f172a; color: white; margin: 0; padding: 0; min-height: 100vh; overflow-x: hidden; }
            h2 { text-align: center; padding: 15px 0; font-weight: 600; text-shadow: 0 2px 10px rgba(99, 102, 241, 0.5); font-size: 1.5rem; }
            .stats { display: flex; justify-content: center; gap: 8px; padding: 0 10px; margin-bottom: 15px; max-width: 900px; margin-left: auto; margin-right: auto; }
            .stat-card { background: var(--card); backdrop-filter: blur(10px); padding: 12px 5px; border-radius: 15px; flex: 1; text-align: center; border: 1px solid var(--glass-border); }
            .stat-label { font-size: 12px; color: #94a3b8; display: block; }
            .stat-value { font-size: 24px; font-weight: 600; }
            #f { color: var(--success); } #r { color: var(--warning); } #o { color: var(--danger); }
            .video-outer-container { width: 95%; max-width: 850px; margin: 0 auto; }
            .video-container { width: 100%; background: #000; border-radius: 15px; overflow: hidden; box-shadow: 0 10px 25px rgba(0,0,0,0.5); border: 2px solid var(--glass-border); }
            img#parking-stream { width: 100%; height: auto; display: block; }
            .panel { background: var(--card); backdrop-filter: blur(12px); padding: 20px; border-radius: 20px; border: 1px solid var(--glass-border); margin: 15px auto; width: 90%; max-width: 500px; }
            .btn { width: 100%; padding: 14px; border-radius: 12px; border: none; font-weight: 600; cursor: pointer; color: white; margin-top: 10px; transition: 0.2s; font-size: 16px; }
            .btn:active { transform: scale(0.97); }
            select { width: 100%; padding: 12px; background: #1e293b; color: white; border-radius: 10px; border: 1px solid var(--glass-border); font-size: 16px; margin-bottom: 5px; }
            .arrival-msg { background: var(--warning); color: #000; padding: 12px; border-radius: 10px; font-weight: 600; margin-bottom: 15px; text-align: center; display: none; }
            
            /* --- [Style สำหรับระบบพับกราฟ] --- */
            .chart-wrapper { width: 90%; max-width: 850px; margin: 10px auto 30px; }
            .chart-toggle-btn { 
                background: rgba(255,255,255,0.05); border: 1px solid var(--glass-border); color: #94a3b8; 
                width: 100%; padding: 10px; border-radius: 12px; cursor: pointer; font-size: 14px; 
                transition: 0.3s; display: flex; justify-content: center; align-items: center; gap: 8px;
            }
            .chart-toggle-btn:hover { background: rgba(255,255,255,0.1); color: white; }
            .chart-container { 
                max-height: 0; overflow: hidden; transition: max-height 0.5s cubic-bezier(0, 1, 0, 1); 
                background: var(--card); margin-top: 10px; border-radius: 15px; border: 0px solid var(--glass-border);
            }
            .chart-container.show { max-height: 1000px; transition: max-height 1s ease-in-out; border: 1px solid var(--glass-border); padding: 15px; }
            .arrow { transition: 0.3s; display: inline-block; }
            .arrow.up { transform: rotate(180deg); }
        </style>
    </head>
    <body>
        <h2>🅿️ Find Parking</h2>
        <div style="text-align:center; margin-bottom:10px;"><button id="audioBtn" class="btn" style="background:var(--primary); max-width:200px; padding: 8px;" onclick="initAudio()">🔊 เปิดเสียง</button></div>
        
        <div class="stats">
            <div class="stat-card"><span class="stat-label">ว่าง</span><span id="f" class="stat-value">0</span></div>
            <div class="stat-card"><span class="stat-label">จอง</span><span id="r" class="stat-value">0</span></div>
            <div class="stat-card"><span class="stat-label">เต็ม</span><span id="o" class="stat-value">0</span></div>
        </div>

        <div class="video-outer-container">
            <div class="video-container"><img id="parking-stream" src="/video_feed"></div>
        </div>

        <div id="bookingPanel" class="panel">
            <select id="slotSelect"></select>
            <button id="reserveBtn" class="btn" style="background:var(--success)" onclick="reserve()">✅ จองตอนนี้</button>
        </div>

        <div id="activePanel" class="panel" style="display:none;">
            <div id="arrivalNotice" class="arrival-msg">✅ ถึงที่จอดรถแล้ว</div>
            <div id="slotLabel" style="color:#94a3b8; text-align:center; margin-bottom:5px;"></div>
            <div id="timeDisplay" style="font-size:40px; font-weight:600; color:var(--warning); text-align:center;">00:00</div>
            <button class="btn" style="background:#8b5cf6;" onclick="extend()">⏳ ต่อเวลา (+90 วิ)</button>
            <a id="navBtn" href="#" target="_blank" class="btn" style="background:#4285F4; text-decoration:none; display:block; text-align:center;">📍 นำทาง</a>
            <button class="btn" style="background:var(--danger)" onclick="cancel()">❌ ยกเลิก</button>
        </div>

        <div class="chart-wrapper">
            <button class="chart-toggle-btn" onclick="toggleChart()">
                📊 <span id="toggleText">แสดงสถิติการใช้งาน</span> <span id="toggleArrow" class="arrow">▼</span>
            </button>
            <div id="chartContainer" class="chart-container">
                <h3 style="text-align:center; margin-top:0; font-size:0.9rem; color:#94a3b8;">ความหนาแน่นรายชั่วโมง (วันนี้ vs เฉลี่ย)</h3>
                <canvas id="parkingChart"></canvas>
            </div>
        </div>

        <script>
            let audioEnabled = false, arrivedAlert = false;
            let parkingChart;
            let isChartVisible = false;

            function initAudio() { audioEnabled = true; document.getElementById('audioBtn').style.display = 'none'; speak("ระบบพร้อมค่ะ"); }
            function speak(t) { if (!audioEnabled) return; window.speechSynthesis.cancel(); const m = new SpeechSynthesisUtterance(t); m.lang = 'th-TH'; window.speechSynthesis.speak(m); }
            
            function toggleChart() {
                const container = document.getElementById('chartContainer');
                const text = document.getElementById('toggleText');
                const arrow = document.getElementById('toggleArrow');
                isChartVisible = !isChartVisible;
                
                if(isChartVisible) {
                    container.classList.add('show');
                    text.innerText = "ซ่อนสถิติการใช้งาน";
                    arrow.classList.add('up');
                    updateChart(); // โหลดข้อมูลทันทีเมื่อกางออก
                } else {
                    container.classList.remove('show');
                    text.innerText = "แสดงสถิติการใช้งาน";
                    arrow.classList.remove('up');
                }
            }

            function updateChart() {
                if(!isChartVisible) return; // ไม่โหลดถ้าพับอยู่
                fetch('/api/hourly_stats').then(r => r.json()).then(data => {
                    const ctx = document.getElementById('parkingChart').getContext('2d');
                    if (parkingChart) {
                        parkingChart.data.datasets[0].data = data.today;
                        parkingChart.data.datasets[1].data = data.average;
                        parkingChart.update();
                    } else {
                        parkingChart = new Chart(ctx, {
                            type: 'line',
                            data: {
                                labels: Array.from({length: 24}, (_, i) => i + ":00"),
                                datasets: [
                                    { label: 'วันนี้', data: data.today, borderColor: '#6366f1', tension: 0.4, fill: true, backgroundColor: 'rgba(99, 102, 241, 0.1)', pointRadius: 2 },
                                    { label: 'ค่าเฉลี่ย', data: data.average, borderColor: '#94a3b8', borderDash: [5, 5], tension: 0.4, fill: false, pointRadius: 0 }
                                ]
                            },
                            options: { 
                                responsive: true, 
                                scales: { 
                                    y: { beginAtZero: true, ticks: { color: '#64748b' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                                    x: { ticks: { color: '#64748b' }, grid: { display: false } }
                                },
                                plugins: { legend: { labels: { color: '#94a3b8', boxWidth: 12 } } }
                            }
                        });
                    }
                });
            }

            function update() {
                const myId = localStorage.getItem('mySlotId');
                fetch('/api/all_data').then(r=>r.json()).then(data => {
                    document.getElementById('f').innerText = data.stats.free;
                    document.getElementById('r').innerText = data.stats.reserved;
                    document.getElementById('o').innerText = data.stats.occupied;
                    const select = document.getElementById('slotSelect');
                    select.innerHTML = data.slots.map(s => `<option value="${s.id}" ${s.status!==0?'disabled':''}>ช่องที่ ${s.id+1} ${s.status==1?'(เต็ม)':s.status==2?'(จอง)':''}</option>`).join('');
                    
                    const mySlot = data.slots.find(s => s.id == myId);
                    if(mySlot && mySlot.status == 2) {
                        document.getElementById('bookingPanel').style.display = 'none';
                        document.getElementById('activePanel').style.display = 'block';
                        document.getElementById('slotLabel').innerText = "คุณจองช่อง " + (parseInt(myId)+1);
                        document.getElementById('navBtn').href = "https://www.google.com/maps?q=" + mySlot.gps;
                        document.getElementById('timeDisplay').innerText = Math.floor(mySlot.remaining/60).toString().padStart(2,'0')+":"+(mySlot.remaining%60).toString().padStart(2,'0');
                        
                        if(mySlot.is_arrived) {
                            document.getElementById('arrivalNotice').style.display = 'block';
                            if(!arrivedAlert) { speak("ถึงที่จอดรถแล้วค่ะ"); arrivedAlert = true; }
                        } else {
                            document.getElementById('arrivalNotice').style.display = 'none';
                            arrivedAlert = false;
                        }
                    } else {
                        document.getElementById('bookingPanel').style.display = 'block';
                        document.getElementById('activePanel').style.display = 'none';
                        document.getElementById('arrivalNotice').style.display = 'none';
                        arrivedAlert = false;
                    }
                });
            }

            function reserve() { 
                const id = document.getElementById('slotSelect').value; 
                fetch('/api/reserve',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({slot_id:id})}).then(r=>r.json()).then(res=>{ if(res.status=='success'){ localStorage.setItem('mySlotId',id); speak("จองสำเร็จ"); update(); } }); 
            }
            function extend() { fetch('/api/extend',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({slot_id:localStorage.getItem('mySlotId')})}).then(r=>r.json()).then(res=>{ if(res.status=='success') speak("ต่อเวลาแล้วค่ะ"); }); }
            function cancel() { 
                fetch('/api/cancel',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({slot_id:localStorage.getItem('mySlotId')})}).then(()=>{ 
                    localStorage.removeItem('mySlotId'); 
                    document.getElementById('arrivalNotice').style.display = 'none';
                    arrivedAlert = false;
                    speak("ยกเลิกแล้ว"); 
                    update(); 
                }); 
            }

            setInterval(update, 1000);
            setInterval(updateChart, 30000); 
            update();
        </script>
    </body>
    </html>
    """

if __name__ == '__main__':
    threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5001, threaded=True, use_reloader=False), daemon=True).start()
    main_process()