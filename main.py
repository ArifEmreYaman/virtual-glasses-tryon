import cv2
import mediapipe as mp
import numpy as np
import open3d as o3d
import copy
import math
from collections import deque

# -----------------------
# AYARLAR
# -----------------------
WIDTH, HEIGHT = 640, 480
MESH_PATH = "glasses.obj"
MESH_BASE_SCALE = 0.01
MODEL_METERS = 0.001

# --- İNCE AYAR PARAMETRELERİ ---
X_FINE_TUNE = 0.0         # Sağa (+) / Sola (-) ince ayar
Y_FINE_TUNE = 0.005       # Aşağı (+) / Yukarı (-) ince ayar
Z_OFFSET = 0.0            # İleri (+) / Geri (-) kaydırır
SCALE_MULTIPLIER = 0.5    # Gözlüğün genel boyutunu ayarlar

# --- GÖZLÜK TİPLERİ VE OFSETLERİ ---
GLASSES_TYPES = {
    'normal': {'y_offset': 0.005, 'z_offset': 0.0, 'scale': 1.0},
    'sunglasses': {'y_offset': 0.008, 'z_offset': 0.002, 'scale': 1.1},
    'reading': {'y_offset': 0.003, 'z_offset': -0.001, 'scale': 0.9},
    'sports': {'y_offset': 0.006, 'z_offset': 0.001, 'scale': 1.05}
}
CURRENT_GLASSES_TYPE = 'normal'  # Varsayılan tip

# --- YENİ: TEMPORAL SMOOTHING PARAMETRELERİ ---
SMOOTHING_ENABLED = True
POSITION_SMOOTHING = 0.7      # 0-1 arası, yüksek değer daha stabil ama daha yavaş
ROTATION_SMOOTHING = 0.6      # Rotasyon için ayrı smoothing
SCALE_SMOOTHING = 0.8         # Scale için smoothing

# --- OTOMATIK KONUMLANDıRMA İÇİN LANDMARK ID'LERİ ---
LEFT_EYE_CORNER_ID = 133
RIGHT_EYE_CORNER_ID = 362
NOSE_TIP_ID = 1
LEFT_EYE_CENTER_ID = 468
RIGHT_EYE_CENTER_ID = 473

# --- DİĞER AYARLAR ---
SCALE_TRANSLATION = 1.0
ALPHA = 0.3                   # Blending alpha değeri
DEBUG_MODE = False            # Debug bilgilerini göster

# -----------------------
# YARDIMCI SINIFLAR
# -----------------------
class PositionSmoother:
    """3D pozisyon ve rotasyon için temporal smoothing"""
    def __init__(self, position_alpha=0.7, rotation_alpha=0.6, scale_alpha=0.8):
        self.pos_alpha = position_alpha
        self.rot_alpha = rotation_alpha
        self.scale_alpha = scale_alpha
        
        self.prev_position = None
        self.prev_rotation = None
        self.prev_scale = None
        
    def smooth_position(self, current_pos):
        if self.prev_position is None:
            self.prev_position = current_pos.copy()
            return current_pos
            
        smoothed = self.pos_alpha * self.prev_position + (1 - self.pos_alpha) * current_pos
        self.prev_position = smoothed.copy()
        return smoothed
        
    def smooth_rotation(self, current_rot):
        if self.prev_rotation is None:
            self.prev_rotation = current_rot.copy()
            return current_rot
            
        smoothed = self.rot_alpha * self.prev_rotation + (1 - self.rot_alpha) * current_rot
        self.prev_rotation = smoothed.copy()
        return smoothed
        
    def smooth_scale(self, current_scale):
        if self.prev_scale is None:
            self.prev_scale = current_scale
            return current_scale
            
        smoothed = self.scale_alpha * self.prev_scale + (1 - self.scale_alpha) * current_scale
        self.prev_scale = smoothed
        return smoothed

class FaceMetrics:
    """Yüz ölçümlerini hesaplayan sınıf"""
    def __init__(self):
        self.face_width_history = deque(maxlen=10)
        
    def calculate_face_width(self, landmarks, width, height):
        """Yüz genişliğini hesapla"""
        left_face = landmarks.landmark[172]  # Sol yanak
        right_face = landmarks.landmark[397]  # Sağ yanak
        
        face_width_px = abs((right_face.x - left_face.x) * width)
        self.face_width_history.append(face_width_px)
        
        # Ortalama al (stabilite için)
        return np.mean(self.face_width_history) if self.face_width_history else face_width_px
    
    def calculate_eye_distance(self, landmarks, width, height):
        """Göz arası mesafeyi hesapla"""
        left_eye = landmarks.landmark[LEFT_EYE_CENTER_ID]
        right_eye = landmarks.landmark[RIGHT_EYE_CENTER_ID]
        
        eye_dist_px = abs((right_eye.x - left_eye.x) * width)
        return eye_dist_px

# -----------------------
# MESH YÜKLEME VE RENDERER AYARLARI
# -----------------------
try:
    mesh_orig = o3d.io.read_triangle_mesh(MESH_PATH)
    if not mesh_orig.has_vertices():
        raise SystemExit(f"Mesh boş veya bulunamadı: {MESH_PATH}")
    
    mesh_orig.compute_vertex_normals()
    mesh_orig.scale(MESH_BASE_SCALE, center=mesh_orig.get_center())
    print(f"Mesh başarıyla yüklendi: {len(mesh_orig.vertices)} vertices")
except Exception as e:
    raise SystemExit(f"Mesh yükleme hatası: {e}")

# Offscreen renderer
renderer = o3d.visualization.rendering.OffscreenRenderer(WIDTH, HEIGHT)
mat = o3d.visualization.rendering.MaterialRecord()
mat.shader = "defaultLit"
mat.base_color = [0.8, 0.8, 0.8, 0.9]  # Hafif şeffaf
cam = renderer.scene.camera
cam.look_at(center=[0, 0, 0], eye=[0, 0, 2], up=[0, 1, 0])

# -----------------------
# MEDIAPIPE AYARLARI
# -----------------------
mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    refine_landmarks=True
)

# -----------------------
# SOLVEPNP MODEL POINTS
# -----------------------
# 3D model noktaları (mm cinsinden, sonra metre'ye çevrilecek)
model_points_mm = np.array([
    (0.0, 0.0, 0.0),        # Burun ucu
    (0.0, -63.6, -12.5),    # Çene
    (-43.3, 32.7, -26.0),   # Sol göz köşesi
    (43.3, 32.7, -26.0),    # Sağ göz köşesi
    (-28.9, -28.9, -20.0),  # Sol ağız köşesi
    (28.9, -28.9, -20.0)    # Sağ ağız köşesi
], dtype=np.float64)

LANDMARK_IDS = [1, 152, 33, 263, 61, 291]

# -----------------------
# KAMERA AYARLARI
# -----------------------
focal_length = WIDTH
center = (WIDTH / 2, HEIGHT / 2)
camera_matrix = np.array([
    [focal_length, 0, center[0]],
    [0, focal_length, center[1]],
    [0, 0, 1]
], dtype=np.float64)
dist_coeffs = np.zeros((4, 1))

# Webcam başlat
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    raise SystemExit("Kamera açılamadı!")

# -----------------------
# YARDIMCI FONKSİYONLAR
# -----------------------
def rodrigues_to_mat(rvec):
    """Rodrigues vektörünü rotasyon matrisine çevir"""
    R, _ = cv2.Rodrigues(rvec)
    return R

def rot2euler(R):
    """Rotasyon matrisini Euler açılarına çevir"""
    sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-6
    
    if not singular:
        x = math.atan2(R[2,1], R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
        
    return np.array([x, y, z])

def euler2rot(euler):
    """Euler açılarını rotasyon matrisine çevir"""
    x, y, z = euler
    
    Rx = np.array([[1, 0, 0],
                   [0, math.cos(x), -math.sin(x)],
                   [0, math.sin(x), math.cos(x)]])
    
    Ry = np.array([[math.cos(y), 0, math.sin(y)],
                   [0, 1, 0],
                   [-math.sin(y), 0, math.cos(y)]])
    
    Rz = np.array([[math.cos(z), -math.sin(z), 0],
                   [math.sin(z), math.cos(z), 0],
                   [0, 0, 1]])
    
    return Rz @ Ry @ Rx

def draw_debug_info(frame, landmarks, target_point_2d, model_origin_2d, face_width, eye_distance):
    """Debug bilgilerini çiz"""
    if not DEBUG_MODE:
        return
        
    # Hedef nokta (yeşil)
    cv2.circle(frame, tuple(target_point_2d.astype(int)), 5, (0, 255, 0), -1)
    
    # Model merkezi (kırmızı)
    cv2.circle(frame, tuple(model_origin_2d.astype(int)), 5, (0, 0, 255), -1)
    
    # Göz noktaları
    left_eye = landmarks.landmark[LEFT_EYE_CENTER_ID]
    right_eye = landmarks.landmark[RIGHT_EYE_CENTER_ID]
    
    left_pt = (int(left_eye.x * WIDTH), int(left_eye.y * HEIGHT))
    right_pt = (int(right_eye.x * WIDTH), int(right_eye.y * HEIGHT))
    
    cv2.circle(frame, left_pt, 3, (255, 0, 0), -1)
    cv2.circle(frame, right_pt, 3, (255, 0, 0), -1)
    cv2.line(frame, left_pt, right_pt, (255, 0, 0), 1)
    
    # Bilgi metinleri
    cv2.putText(frame, f"Face Width: {face_width:.1f}px", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"Eye Distance: {eye_distance:.1f}px", (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"Glasses Type: {CURRENT_GLASSES_TYPE}", (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

def draw_axis(frame, camera_matrix, dist_coeffs, rvec, tvec, length=50):
    """3D eksen çizimi"""
    axis3d = np.float32([[length,0,0],[0,length,0],[0,0,length]]).reshape(-1,3)
    imgpts, _ = cv2.projectPoints(axis3d, rvec, tvec, camera_matrix, dist_coeffs)
    corner, _ = cv2.projectPoints(np.zeros((1,3)), rvec, tvec, camera_matrix, dist_coeffs)
    
    corner = tuple(corner.ravel().astype(int))
    imgpts = imgpts.reshape(-1,2).astype(int)
    
    # X ekseni (kırmızı), Y ekseni (yeşil), Z ekseni (mavi)
    cv2.line(frame, corner, tuple(imgpts[0]), (0,0,255), 3)
    cv2.line(frame, corner, tuple(imgpts[1]), (0,255,0), 3)
    cv2.line(frame, corner, tuple(imgpts[2]), (255,0,0), 3)

def calculate_optimal_position(landmarks, width, height):
    """Optimal gözlük pozisyonunu hesapla"""
    # Göz köşelerini al
    left_eye_corner = landmarks.landmark[LEFT_EYE_CORNER_ID]
    right_eye_corner = landmarks.landmark[RIGHT_EYE_CORNER_ID]
    
    # Göz merkezlerini al (daha hassas)
    left_eye_center = landmarks.landmark[LEFT_EYE_CENTER_ID]
    right_eye_center = landmarks.landmark[RIGHT_EYE_CENTER_ID]
    
    # Hedef nokta: Gözlerin ortası (biraz daha yukarıda)
    target_x = (left_eye_center.x + right_eye_center.x) / 2
    target_y = (left_eye_center.y + right_eye_center.y) / 2 - 0.01  # Biraz yukarı
    
    target_point_2d = np.array([target_x * width, target_y * height], dtype=np.float64)
    
    return target_point_2d

def get_adaptive_scale(eye_distance, face_width, base_scale=1.0):
    """Yüz boyutuna göre adaptif ölçek hesapla"""
    # Standart değerler (ortalama yetişkin için)
    standard_eye_distance = 65.0  # piksel
    standard_face_width = 120.0   # piksel
    
    # Ölçek faktörlerini hesapla
    eye_scale_factor = eye_distance / standard_eye_distance
    face_scale_factor = face_width / standard_face_width
    
    # İki faktörün ortalamasını al ve limitle
    combined_scale = (eye_scale_factor + face_scale_factor) / 2
    combined_scale = np.clip(combined_scale, 0.5, 2.0)  # Aşırı büyük/küçük önle
    
    return base_scale * combined_scale

# -----------------------
# ANA DÖNGÜ
# -----------------------
print("AR Gözlük Uygulaması Başlatılıyor...")
print("Tuşlar:")
print("  ESC: Çıkış")
print("  D: Debug modu aç/kapa")
print("  1-4: Gözlük tipi değiştir (normal/sunglasses/reading/sports)")
print("  S: Smoothing aç/kapa")

# Yardımcı sınıfları başlat
smoother = PositionSmoother(POSITION_SMOOTHING, ROTATION_SMOOTHING, SCALE_SMOOTHING)
face_metrics = FaceMetrics()

frame_count = 0
fps_counter = 0
import time
start_time = time.time()

# Pencereyi önceden oluştur (tek sefer)
cv2.namedWindow("AR Gözlük", cv2.WINDOW_AUTOSIZE)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Kamera frame'i okunamadı!")
            break
            
        frame = cv2.flip(frame, 1)  # Ayna efekti
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # MediaPipe ile yüz tespiti
        results = face_mesh.process(img_rgb)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            
            # SolvePnP için image points
            image_points = np.array([
                (face_landmarks.landmark[idx].x * WIDTH, 
                 face_landmarks.landmark[idx].y * HEIGHT) 
                for idx in LANDMARK_IDS
            ], dtype=np.float64)
            
            model_points = model_points_mm * MODEL_METERS
            
            # Pose estimation
            success, rvec, tvec = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs, 
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if not success:
                # Pencere adı aynı olduğu için yeni pencere açmaz
                frame_count += 1
                key = cv2.waitKey(1) & 0xFF
                if key == 27: break
                continue
            
            # Yüz metrikleri hesapla
            face_width = face_metrics.calculate_face_width(face_landmarks, WIDTH, HEIGHT)
            eye_distance = face_metrics.calculate_eye_distance(face_landmarks, WIDTH, HEIGHT)
            
            # Optimal pozisyonu hesapla
            target_point_2d = calculate_optimal_position(face_landmarks, WIDTH, HEIGHT)
            
            # Mevcut model merkezi
            model_origin_3d = np.array([[0.0, 0.0, 0.0]])
            model_origin_2d, _ = cv2.projectPoints(
                model_origin_3d, rvec, tvec, camera_matrix, dist_coeffs
            )
            model_origin_2d = model_origin_2d.flatten()
            
            # 2D hata hesapla ve 3D'ye çevir
            error_px = target_point_2d - model_origin_2d
            face_depth_m = abs(tvec[2, 0])
            
            fx = camera_matrix[0, 0]
            fy = camera_matrix[1, 1]
            
            offset_x_m = (error_px[0] * face_depth_m) / fx
            offset_y_m = (error_px[1] * face_depth_m) / fy
            
            # Rotasyon matrisini hesapla ve düzelt
            R = rodrigues_to_mat(rvec)
            euler = rot2euler(R)
            euler[1] = -euler[1]  # Y eksenini tersine çevir
            
            # Smoothing uygula
            if SMOOTHING_ENABLED:
                euler = smoother.smooth_rotation(euler)
                offset_x_m = smoother.smooth_position(np.array([offset_x_m]))[0]
                offset_y_m = smoother.smooth_position(np.array([offset_y_m]))[0]
            
            R = euler2rot(euler)
            
            # Transformasyon matrisini oluştur
            t = tvec.reshape(3) * SCALE_TRANSLATION
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R
            T[:3, 3] = t
            
            # Mesh kopyala ve transform et
            mesh = copy.deepcopy(mesh_orig)
            
            # Koordinat sistemi düzeltmesi
            R_fix = np.array([[-1,0,0,0], [0,-1,0,0], [0,0,1,0], [0,0,0,1]], dtype=np.float64)
            mesh.transform(R_fix)
            
            # Ana transformasyon + ofsetler
            T_apply = T.copy()
            
            # Gözlük tipi ofsetleri
            glasses_config = GLASSES_TYPES[CURRENT_GLASSES_TYPE]
            
            T_apply[0,3] += offset_x_m + X_FINE_TUNE
            T_apply[1,3] += offset_y_m + Y_FINE_TUNE + glasses_config['y_offset']
            T_apply[2,3] += Z_OFFSET + glasses_config['z_offset']
            
            mesh.transform(T_apply)
            
            # Adaptif ölçeklendirme
            adaptive_scale = get_adaptive_scale(eye_distance, face_width)
            final_scale = adaptive_scale * SCALE_MULTIPLIER * glasses_config['scale']
            
            if SMOOTHING_ENABLED:
                final_scale = smoother.smooth_scale(final_scale)
            
            mesh.scale(final_scale, center=mesh.get_center())
            
            # Render
            renderer.scene.clear_geometry()
            renderer.scene.add_geometry("mesh", mesh, mat)
            img_o3d = renderer.render_to_image()
            img_o3d = np.asarray(img_o3d)
            img_o3d = cv2.cvtColor(img_o3d, cv2.COLOR_RGB2BGR)
            
            # Blend
            blended = cv2.addWeighted(frame, 1-ALPHA, img_o3d, ALPHA, 0)
            
            # Debug bilgileri
            draw_debug_info(blended, face_landmarks, target_point_2d, 
                          model_origin_2d, face_width, eye_distance)
            
            if DEBUG_MODE:
                draw_axis(blended, camera_matrix, dist_coeffs, rvec, tvec)
            
            # Tek pencereye göster
            cv2.imshow("AR Gözlük", blended)
        else:
            # Yüz tespit edilmediğinde de aynı pencereyi kullan
            cv2.imshow("AR Gözlük", frame)
        
        # FPS hesapla
        frame_count += 1
        if frame_count % 30 == 0:
            current_time = time.time()
            fps = 30 / (current_time - start_time)
            start_time = current_time
            if DEBUG_MODE:
                print(f"FPS: {fps:.1f}")
        
        # Tuş kontrolü
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('d') or key == ord('D'):
            DEBUG_MODE = not DEBUG_MODE
            print(f"Debug modu: {'Açık' if DEBUG_MODE else 'Kapalı'}")
        elif key == ord('s') or key == ord('S'):
            SMOOTHING_ENABLED = not SMOOTHING_ENABLED
            print(f"Smoothing: {'Açık' if SMOOTHING_ENABLED else 'Kapalı'}")
        elif key == ord('1'):
            CURRENT_GLASSES_TYPE = 'normal'
            print("Gözlük tipi: Normal")
        elif key == ord('2'):
            CURRENT_GLASSES_TYPE = 'sunglasses'
            print("Gözlük tipi: Güneş Gözlüğü")
        elif key == ord('3'):
            CURRENT_GLASSES_TYPE = 'reading'
            print("Gözlük tipi: Okuma Gözlüğü")
        elif key == ord('4'):
            CURRENT_GLASSES_TYPE = 'sports'
            print("Gözlük tipi: Spor Gözlüğü")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Uygulama kapatıldı.")
