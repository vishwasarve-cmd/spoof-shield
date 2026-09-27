SPOOFSHIELD WEB EDITION

1. Open a terminal in this folder.
2. Install dependencies:
   pip install -r requirements.txt
3. Run:
   uvicorn app:app --reload
4. Open:
   http://127.0.0.1:8000

The browser handles webcam capture. The Python backend performs:
- MobileNetV2 feature extraction
- MediaPipe face mesh
- EAR blink detection
- MAR smile/open-mouth detection
- 3D pose estimation
- Texture/chrominance analysis
- Weighted ensemble scoring
- SQLite authentication
- CSV telemetry logging

For a production deployment, use HTTPS and proper server-side session/token authentication.
