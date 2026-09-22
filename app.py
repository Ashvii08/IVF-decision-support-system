from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_file
import json, os, pickle, re, io, base64, socket, sqlite3, uuid, logging, csv
from datetime import datetime
import numpy as np
import pandas as pd
from PIL import Image
import pytesseract
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.units import inch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ferticare.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

app = Flask(__name__)
app.secret_key = 'ivf_clinical_secret_2024'

BASE = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(BASE, 'data', 'ivf.db')
UPLOAD_FOLDER = os.path.join(BASE, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.join(BASE, 'data'), exist_ok=True)

try:
    response_model = pickle.load(open(os.path.join(BASE,'models','response_model.pkl'),'rb'))
    ohss_model = pickle.load(open(os.path.join(BASE,'models','ohss_model.pkl'),'rb'))
    label_encoder = pickle.load(open(os.path.join(BASE,'models','label_encoder.pkl'),'rb'))
    features_list = pickle.load(open(os.path.join(BASE,'models','features.pkl'),'rb'))
    feat_importance = pickle.load(open(os.path.join(BASE,'models','feature_importances.pkl'),'rb'))
    MODELS_LOADED = True
except:
    MODELS_LOADED = False
    feat_importance = {}

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return '127.0.0.1'

def generate_patient_id():
    return 'FERT' + str(uuid.uuid4().hex[:8]).upper()

def log_action(action, patient_id=None, details=None):
    username = session.get('user', {}).get('username', 'unknown')
    log_entry = f"User: {username} | Action: {action}"
    if patient_id:
        log_entry += f" | Patient: {patient_id}"
    if details:
        log_entry += f" | Details: {details}"
    logger.info(log_entry)
    try:
        conn = get_db()
        conn.execute('INSERT INTO activity_logs (username, action, patient_id, details) VALUES (?,?,?,?)',
                     (username, action, patient_id or '', details or ''))
        conn.commit()
        conn.close()
    except:
        pass

def get_db():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        role TEXT DEFAULT 'Doctor',
        name TEXT NOT NULL,
        hospital_name TEXT DEFAULT '',
        specialization TEXT DEFAULT '',
        license_number TEXT DEFAULT '',
        department TEXT DEFAULT '',
        contact TEXT DEFAULT ''
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS patients (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        owner TEXT NOT NULL,
        Patient_ID TEXT UNIQUE NOT NULL,
        Patient_Name TEXT DEFAULT '',
        Age INTEGER DEFAULT 0,
        BMI REAL DEFAULT 0,
        AMH REAL DEFAULT 0,
        AFC INTEGER DEFAULT 0,
        Basal_FSH REAL DEFAULT 0,
        Basal_LH REAL DEFAULT 0,
        Basal_E2 REAL DEFAULT 0,
        Stimulation_Days INTEGER DEFAULT 0,
        Dominant_Follicle_Size REAL DEFAULT 0,
        E2_Trigger REAL DEFAULT 0,
        Oocytes_Retrieved INTEGER DEFAULT 0,
        Ovarian_Response TEXT DEFAULT 'Normal',
        OHSS_Risk INTEGER DEFAULT 0,
        Pregnancy INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id TEXT NOT NULL,
        owner TEXT NOT NULL,
        response TEXT,
        ohss_risk INTEGER,
        response_confidence REAL,
        ohss_confidence REAL,
        explanations TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(patient_id) REFERENCES patients(Patient_ID)
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS explanations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id TEXT NOT NULL,
        owner TEXT NOT NULL,
        explanation_text TEXT,
        saved BOOLEAN DEFAULT 1,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(patient_id) REFERENCES patients(Patient_ID)
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS scan_images (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        owner TEXT NOT NULL,
        patient_id TEXT NOT NULL,
        filename TEXT NOT NULL,
        image_data TEXT NOT NULL,
        uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(patient_id) REFERENCES patients(Patient_ID)
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS activity_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        action TEXT,
        patient_id TEXT,
        details TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    if c.execute('SELECT COUNT(*) FROM users').fetchone()[0] == 0:
        c.execute("INSERT INTO users (username,password,role,name,hospital_name,specialization,license_number) VALUES (?,?,?,?,?,?,?)",
                  ('doctor1','doctor123','Doctor','Dr. Priya Sharma','City Fertility Hospital','Reproductive Endocrinology','MCI-12345'))
        c.execute("INSERT INTO users (username,password,role,name,hospital_name,department,contact) VALUES (?,?,?,?,?,?,?)",
                  ('hospital1','hosp123','Hospital','Fertility Care Clinic','Fertility Care Clinic','IVF & ART Centre','+91-9876543210'))
    conn.commit()
    conn.close()

init_db()

def get_db_user(username):
    conn = get_db()
    row = conn.execute('SELECT * FROM users WHERE username=?',(username,)).fetchone()
    conn.close()
    return dict(row) if row else None

def load_patients():
    owner = session.get('user',{}).get('username','')
    conn = get_db()
    rows = conn.execute('SELECT * FROM patients WHERE owner=? ORDER BY created_at DESC',(owner,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_patient(patient_id):
    owner = session.get('user',{}).get('username','')
    conn = get_db()
    row = conn.execute('SELECT * FROM patients WHERE Patient_ID=? AND owner=?',(patient_id, owner)).fetchone()
    conn.close()
    return dict(row) if row else None

def save_new_patient(p):
    owner = session.get('user',{}).get('username','')
    pid = generate_patient_id()
    conn = get_db()
    conn.execute('''INSERT INTO patients
        (owner,Patient_ID,Patient_Name,Age,BMI,AMH,AFC,Basal_FSH,Basal_LH,Basal_E2,
         Stimulation_Days,Dominant_Follicle_Size,E2_Trigger,Oocytes_Retrieved,
         Ovarian_Response,OHSS_Risk,Pregnancy)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',(
        owner, pid,
        p.get('Patient_Name','Unknown'),
        int(float(p.get('Age',0) or 0)),
        float(p.get('BMI',0) or 0),
        float(p.get('AMH',0) or 0),
        int(float(p.get('AFC',0) or 0)),
        float(p.get('Basal_FSH',0) or 0),
        float(p.get('Basal_LH',0) or 0),
        float(p.get('Basal_E2',0) or 0),
        int(float(p.get('Stimulation_Days',0) or 0)),
        float(p.get('Dominant_Follicle_Size',0) or 0),
        float(p.get('E2_Trigger',0) or 0),
        int(float(p.get('Oocytes_Retrieved',0) or 0)),
        p.get('Ovarian_Response','Normal'),
        int(p.get('OHSS_Risk',0) or 0),
        int(p.get('Pregnancy',0) or 0)
    ))
    conn.commit()
    conn.close()
    log_action('Add Patient', pid, p.get('Patient_Name'))
    return pid

def update_patient_db(patient_id, data):
    owner = session.get('user',{}).get('username','')
    allowed = ['Patient_Name','Age','BMI','AMH','AFC','Basal_FSH','Basal_LH','Basal_E2',
               'Stimulation_Days','Dominant_Follicle_Size','E2_Trigger','Oocytes_Retrieved',
               'Ovarian_Response','OHSS_Risk','Pregnancy']
    fields = {}
    for k,v in data.items():
        if k not in allowed: continue
        if k == 'Age': fields[k] = int(float(v or 0))
        elif k in ('AFC','Stimulation_Days','Oocytes_Retrieved','OHSS_Risk','Pregnancy'):
            fields[k] = int(float(v or 0))
        elif k == 'Patient_Name': fields[k] = str(v)
        elif k == 'Ovarian_Response': fields[k] = str(v)
        else: fields[k] = float(v or 0)
    if not fields: return
    fields['updated_at'] = datetime.now().isoformat()
    set_clause = ', '.join(f'{k}=?' for k in fields)
    values = list(fields.values()) + [patient_id, owner]
    conn = get_db()
    conn.execute(f'UPDATE patients SET {set_clause} WHERE Patient_ID=? AND owner=?', values)
    conn.commit()
    conn.close()
    log_action('Update Patient', patient_id)

def delete_patient_db(patient_id):
    owner = session.get('user',{}).get('username','')
    conn = get_db()
    conn.execute('DELETE FROM patients WHERE Patient_ID=? AND owner=?',(patient_id, owner))
    conn.execute('DELETE FROM scan_images WHERE patient_id=? AND owner=?',(patient_id, owner))
    conn.execute('DELETE FROM predictions WHERE patient_id=?',(patient_id,))
    conn.execute('DELETE FROM explanations WHERE patient_id=?',(patient_id,))
    conn.commit()
    conn.close()
    log_action('Delete Patient', patient_id)

def save_prediction(patient_id, pred_data):
    owner = session.get('user',{}).get('username','')
    conn = get_db()
    conn.execute('''INSERT INTO predictions
        (patient_id,owner,response,ohss_risk,response_confidence,ohss_confidence,explanations)
        VALUES (?,?,?,?,?,?,?)''',
        (patient_id, owner, pred_data.get('response'), pred_data.get('ohss_risk'),
         pred_data.get('response_confidence'), pred_data.get('ohss_confidence'),
         json.dumps(pred_data.get('explanations', []))))
    conn.commit()
    conn.close()

def save_explanation(patient_id, explanation_text):
    owner = session.get('user',{}).get('username','')
    conn = get_db()
    conn.execute('''INSERT INTO explanations (patient_id,owner,explanation_text,saved)
        VALUES (?,?,?,1)''', (patient_id, owner, explanation_text))
    conn.commit()
    conn.close()
    log_action('Save Explanation', patient_id)

def get_explanations(patient_id):
    owner = session.get('user',{}).get('username','')
    conn = get_db()
    rows = conn.execute('SELECT * FROM explanations WHERE patient_id=? AND owner=? ORDER BY created_at DESC',
                       (patient_id, owner)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

PARAM_RANGES = {
    'Age': (18, 45),
    'BMI': (18.5, 32),
    'AMH': (1.0, 6.0),
    'AFC': (5, 30),
    'Basal_FSH': (2, 10),
    'Basal_LH': (1, 10),
    'Basal_E2': (20, 80),
    'Stimulation_Days': (8, 14),
    'Dominant_Follicle_Size': (17, 22),
    'E2_Trigger': (1000, 4000),
}

def check_params(data):
    warnings = {}
    for param,(lo,hi) in PARAM_RANGES.items():
        v = data.get(param)
        if v is None: continue
        try:
            fv = float(v)
            if fv > 0 and (fv < lo or fv > hi):
                warnings[param] = {'value':fv,'min':lo,'max':hi,'status':'low' if fv<lo else 'high'}
        except: pass
    return warnings

def smart_autofill(data):
    filled = {}
    filled_explanation = {}
    age = float(data.get('Age') or 0)
    afc = float(data.get('AFC') or 0)
    amh = float(data.get('AMH') or 0)
    fsh = float(data.get('Basal_FSH') or 0)
    if not data.get('AMH') or amh == 0:
        if age > 0 and afc > 0:
            est = max(0.1, round((-0.065 * age + 0.09 * afc + 2.8), 2))
            filled['AMH'] = est
            filled_explanation['AMH'] = f"From Age ({int(age)}) + AFC ({int(afc)})"
        elif age > 0:
            est = max(0.1, round((8.0 - 0.12 * age), 2))
            filled['AMH'] = est
            filled_explanation['AMH'] = f"From Age ({int(age)})"
    if not data.get('Basal_FSH') or fsh == 0:
        if age > 0:
            est = round(min(max(3.0, 0.18 * age + 0.5), 20.0), 2)
            filled['Basal_FSH'] = est
            filled_explanation['Basal_FSH'] = f"From Age ({int(age)})"
    if not data.get('E2_Trigger') or float(data.get('E2_Trigger') or 0) == 0:
        use_afc = afc if afc > 0 else 10
        use_amh = amh if amh > 0 else (filled.get('AMH', 2.0))
        est = round(min(max(800, use_afc * 85 + use_amh * 180), 7000), 0)
        filled['E2_Trigger'] = est
        filled_explanation['E2_Trigger'] = f"From AFC + AMH"
    if not data.get('Basal_LH') or float(data.get('Basal_LH') or 0) == 0:
        if afc > 20:
            filled['Basal_LH'] = round(min(max(5.0, afc * 0.3), 15.0), 2)
            filled_explanation['Basal_LH'] = f"From high AFC"
    return filled, filled_explanation

def ml_predict(data):
    if not MODELS_LOADED:
        return "Normal", 0, 0.75, 0.80, ["AI Models not loaded"]
    try:
        feat_vals = [float(data.get(f,0) or 0) for f in features_list]
        X = pd.DataFrame([dict(zip(features_list, feat_vals))])
        resp_idx = response_model.predict(X)[0]
        resp_proba = response_model.predict_proba(X)[0]
        ohss_pred = ohss_model.predict(X)[0]
        ohss_proba = ohss_model.predict_proba(X)[0]
        response = label_encoder.inverse_transform([resp_idx])[0]
        return response, int(ohss_pred), float(max(resp_proba)), float(max(ohss_proba)), []
    except Exception as e:
        return "Normal", 0, 0.75, 0.80, [f"Prediction error: {str(e)}"]

CSV_COLUMN_MAP = {
    'patient_name': 'Patient_Name', 'name': 'Patient_Name',
    'age': 'Age',
    'bmi': 'BMI',
    'amh': 'AMH',
    'afc': 'AFC',
    'basal_fsh': 'Basal_FSH', 'fsh': 'Basal_FSH',
    'basal_lh': 'Basal_LH', 'lh': 'Basal_LH',
    'basal_e2': 'Basal_E2', 'e2': 'Basal_E2',
    'stimulation_days': 'Stimulation_Days', 'stim_days': 'Stimulation_Days',
    'dominant_follicle_size': 'Dominant_Follicle_Size', 'follicle_size': 'Dominant_Follicle_Size',
    'e2_trigger': 'E2_Trigger',
    'oocytes_retrieved': 'Oocytes_Retrieved', 'oocytes': 'Oocytes_Retrieved',
    'ovarian_response': 'Ovarian_Response',
    'ohss_risk': 'OHSS_Risk',
    'pregnancy': 'Pregnancy',
}

@app.route('/')
def index():
    return redirect(url_for('overview') if 'user' in session else url_for('login'))

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        d = request.get_json()
        username = (d.get('username','') or '').strip()
        password = (d.get('password','') or '').strip()
        conn = get_db()
        row = conn.execute('SELECT * FROM users WHERE username=? AND password=?',(username,password)).fetchone()
        conn.close()
        if row:
            session['user'] = dict(row)
            log_action('Login', details=f"Role: {row['role']}")
            return jsonify({'success':True, 'role': row['role']})
        log_action('Failed Login', details=username)
        return jsonify({'success':False,'message':'Invalid credentials'})
    return render_template('login.html')

@app.route('/register', methods=['POST'])
def register():
    d = request.get_json()
    username = (d.get('username','') or '').strip()
    password = (d.get('password','') or '').strip()
    role = d.get('role','Doctor')
    name = (d.get('name','') or '').strip()
    hospital = (d.get('hospital_name','') or '').strip()
    spec = (d.get('specialization','') or '').strip()
    lic = (d.get('license_number','') or '').strip()
    dept = (d.get('department','') or '').strip()
    contact = (d.get('contact','') or '').strip()
    if not username or not password or not name:
        return jsonify({'success':False,'message':'All required fields must be filled'})
    conn = get_db()
    try:
        conn.execute('''INSERT INTO users (username,password,role,name,hospital_name,specialization,license_number,department,contact)
                        VALUES (?,?,?,?,?,?,?,?,?)''',
                     (username,password,role,name,hospital,spec,lic,dept,contact))
        conn.commit()
        conn.close()
        log_action('Register', details=f"{name} ({role})")
        return jsonify({'success':True})
    except sqlite3.IntegrityError:
        conn.close()
        return jsonify({'success':False,'message':'Username already exists'})

@app.route('/logout')
def logout():
    log_action('Logout')
    session.clear()
    return redirect(url_for('login'))

@app.route('/overview')
def overview():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('overview.html', user=session['user'])

@app.route('/dashboard/<patient_id>')
def dashboard(patient_id):
    if 'user' not in session:
        return redirect(url_for('login'))
    patient = get_patient(patient_id)
    if not patient:
        return redirect(url_for('overview'))
    return render_template('dashboard.html', patient=patient, user=session['user'])

@app.route('/dashboard/<patient_id>/clinical-data')
def clinical_data(patient_id):
    if 'user' not in session:
        return redirect(url_for('login'))
    patient = get_patient(patient_id)
    if not patient:
        return redirect(url_for('overview'))
    return render_template('clinical_data.html', patient=patient, user=session['user'])

@app.route('/dashboard/<patient_id>/simulation')
def simulation(patient_id):
    if 'user' not in session:
        return redirect(url_for('login'))
    patient = get_patient(patient_id)
    if not patient:
        return redirect(url_for('overview'))
    return render_template('simulation.html', patient=patient, user=session['user'])

@app.route('/dashboard/<patient_id>/ai-prediction')
def ai_prediction(patient_id):
    if 'user' not in session:
        return redirect(url_for('login'))
    patient = get_patient(patient_id)
    if not patient:
        return redirect(url_for('overview'))
    return render_template('ai_prediction.html', patient=patient, user=session['user'])

@app.route('/dashboard/<patient_id>/ai-explanation')
def ai_explanation(patient_id):
    if 'user' not in session:
        return redirect(url_for('login'))
    patient = get_patient(patient_id)
    if not patient:
        return redirect(url_for('overview'))
    explanations = get_explanations(patient_id)
    return render_template('ai_explanation.html', patient=patient, user=session['user'], explanations=explanations)

@app.route('/dashboard/<patient_id>/scan-reports')
def scan_reports(patient_id):
    if 'user' not in session:
        return redirect(url_for('login'))
    patient = get_patient(patient_id)
    if not patient:
        return redirect(url_for('overview'))
    return render_template('scan_reports.html', patient=patient, user=session['user'])

@app.route('/api/patients')
def api_patients():
    if 'user' not in session: return jsonify({'error':'Unauthorized'}),401
    return jsonify(load_patients())

@app.route('/api/patient/<patient_id>')
def api_patient(patient_id):
    if 'user' not in session: return jsonify({'error':'Unauthorized'}),401
    patient = get_patient(patient_id)
    if not patient: return jsonify({'error':'Not found'}),404
    return jsonify(patient)

@app.route('/api/add_patient', methods=['POST'])
def add_patient():
    if 'user' not in session: return jsonify({'error':'Unauthorized'}),401
    data = request.get_json()
    pid = save_new_patient(data)
    return jsonify({'success':True,'patient_id':pid})

@app.route('/api/update_patient/<patient_id>', methods=['POST'])
def update_patient(patient_id):
    if 'user' not in session: return jsonify({'error':'Unauthorized'}),401
    data = request.get_json()
    update_patient_db(patient_id, data)
    return jsonify({'success':True})

@app.route('/api/delete_patient/<patient_id>', methods=['DELETE'])
def delete_patient(patient_id):
    if 'user' not in session: return jsonify({'error':'Unauthorized'}),401
    delete_patient_db(patient_id)
    return jsonify({'success':True})

@app.route('/api/import_file', methods=['POST'])
def import_file():
    if 'user' not in session: return jsonify({'error':'Unauthorized'}),401
    if 'file' not in request.files:
        return jsonify({'success':False,'message':'No file uploaded'}),400
    file = request.files['file']
    fname = file.filename.lower()
    
    rows_data = []
    
    try:
        if fname.endswith('.csv'):
            stream = io.StringIO(file.stream.read().decode('utf-8-sig'))
            reader = csv.DictReader(stream)
            for row in reader:
                rows_data.append(dict(row))
        elif fname.endswith('.xlsx') or fname.endswith('.xls'):
            try:
                import openpyxl
                wb = openpyxl.load_workbook(io.BytesIO(file.stream.read()), data_only=True)
                ws = wb.active
                headers = []
                for i, row in enumerate(ws.iter_rows(values_only=True)):
                    if i == 0:
                        headers = [str(c).strip() if c is not None else '' for c in row]
                    else:
                        row_dict = {}
                        for j, val in enumerate(row):
                            if j < len(headers):
                                row_dict[headers[j]] = str(val).strip() if val is not None else ''
                        rows_data.append(row_dict)
            except ImportError:
                return jsonify({'success':False,'message':'openpyxl not installed. Run: pip install openpyxl'}),500
        else:
            return jsonify({'success':False,'message':'Only CSV and Excel (.xlsx/.xls) files are supported'}),400

        imported = 0
        skipped = 0
        errors = []
        for row_num, row in enumerate(rows_data, start=2):
            try:
                mapped = {}
                for col, val in row.items():
                    key = col.strip().lower().replace(' ','_')
                    if key in CSV_COLUMN_MAP:
                        mapped[CSV_COLUMN_MAP[key]] = str(val).strip() if val else ''
                if not mapped.get('Patient_Name') or mapped.get('Patient_Name') in ('','None','nan'):
                    skipped += 1
                    errors.append(f"Row {row_num}: Missing patient name, skipped")
                    continue
                save_new_patient(mapped)
                imported += 1
            except Exception as e:
                skipped += 1
                errors.append(f"Row {row_num}: {str(e)}")
        
        log_action('File Import', details=f"Imported: {imported}, Skipped: {skipped}")
        return jsonify({
            'success': True,
            'imported': imported,
            'skipped': skipped,
            'errors': errors[:10]
        })
    except Exception as e:
        return jsonify({'success':False,'message':f'Parse error: {str(e)}'}),500

@app.route('/api/csv_template')
def csv_template():
    if 'user' not in session: return jsonify({'error':'Unauthorized'}),401
    headers = ['Patient_Name','Age','BMI','AMH','AFC','Basal_FSH','Basal_LH','Basal_E2',
               'Stimulation_Days','Dominant_Follicle_Size','E2_Trigger','Oocytes_Retrieved',
               'Ovarian_Response','OHSS_Risk','Pregnancy']
    sample = ['Jane Doe','32','22.5','2.1','12','6.5','5.2','45.0','10','19.5','2500','8','Normal','0','0']
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(headers)
    writer.writerow(sample)
    buf.seek(0)
    return send_file(
        io.BytesIO(buf.getvalue().encode()),
        as_attachment=True,
        download_name='ferticare_import_template.csv',
        mimetype='text/csv'
    )

@app.route('/api/xlsx_template')
def xlsx_template():
    if 'user' not in session: return jsonify({'error':'Unauthorized'}),401
    try:
        import openpyxl
        from openpyxl.styles import Font, PatternFill, Alignment
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "FertiCare Import"
        headers = ['Patient_Name','Age','BMI','AMH','AFC','Basal_FSH','Basal_LH','Basal_E2',
                   'Stimulation_Days','Dominant_Follicle_Size','E2_Trigger','Oocytes_Retrieved',
                   'Ovarian_Response','OHSS_Risk','Pregnancy']
        sample = ['Jane Doe',32,22.5,2.1,12,6.5,5.2,45.0,10,19.5,2500,8,'Normal',0,0]
        for col, h in enumerate(headers, 1):
            cell = ws.cell(row=1, column=col, value=h)
            cell.font = Font(bold=True, color='FFFFFF')
            cell.fill = PatternFill(start_color='9B72C8', end_color='9B72C8', fill_type='solid')
            cell.alignment = Alignment(horizontal='center')
            ws.column_dimensions[cell.column_letter].width = max(len(h)+4, 14)
        for col, v in enumerate(sample, 1):
            ws.cell(row=2, column=col, value=v)
        buf = io.BytesIO()
        wb.save(buf)
        buf.seek(0)
        return send_file(buf, as_attachment=True, download_name='ferticare_import_template.xlsx',
                        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    except ImportError:
        return jsonify({'error':'openpyxl not installed'}),500

@app.route('/api/get_scans/<patient_id>')
def get_scans(patient_id):
    if 'user' not in session: return jsonify({'error':'Unauthorized'}),401
    owner = session['user']['username']
    conn = get_db()
    rows = conn.execute(
        'SELECT id,filename,image_data,uploaded_at FROM scan_images WHERE patient_id=? AND owner=? ORDER BY uploaded_at DESC',
        (patient_id, owner)
    ).fetchall()
    conn.close()
    return jsonify({'scans': [dict(r) for r in rows]})

@app.route('/api/save_scan', methods=['POST'])
def save_scan():
    if 'user' not in session: return jsonify({'error':'Unauthorized'}),401
    owner = session['user']['username']
    d = request.get_json()
    conn = get_db()
    conn.execute('INSERT INTO scan_images (owner,patient_id,filename,image_data) VALUES (?,?,?,?)',
                 (owner, d.get('patient_id',''), d.get('filename',''), d.get('image_b64','')))
    conn.commit()
    conn.close()
    log_action('Upload Scan', d.get('patient_id'))
    return jsonify({'success':True})

@app.route('/api/delete_scan/<int:scan_id>', methods=['DELETE'])
def delete_scan(scan_id):
    if 'user' not in session: return jsonify({'error':'Unauthorized'}),401
    owner = session['user']['username']
    conn = get_db()
    conn.execute('DELETE FROM scan_images WHERE id=? AND owner=?',(scan_id,owner))
    conn.commit()
    conn.close()
    log_action('Delete Scan', details=f"ID: {scan_id}")
    return jsonify({'success':True})

@app.route('/api/check_params', methods=['POST'])
def api_check_params():
    if 'user' not in session: return jsonify({'error':'Unauthorized'}),401
    data = request.get_json()
    warnings = check_params(data)
    return jsonify({'warnings': warnings})

@app.route('/api/autofill', methods=['POST'])
def api_autofill():
    if 'user' not in session: return jsonify({'error':'Unauthorized'}),401
    data = request.get_json()
    filled, explanations = smart_autofill(data)
    return jsonify({'success':True, 'filled':filled, 'explanations':explanations})

@app.route('/api/predict', methods=['POST'])
def api_predict():
    if 'user' not in session: return jsonify({'error':'Unauthorized'}),401
    data = request.get_json()
    response, ohss, resp_conf, ohss_conf, exps = ml_predict(data)
    save_prediction(data.get('Patient_ID'), {
        'response': response,
        'ohss_risk': ohss,
        'response_confidence': resp_conf,
        'ohss_confidence': ohss_conf,
        'explanations': exps
    })
    fi = {k:round(float(v)*100,1) for k,v in feat_importance.items()} if feat_importance else {}
    return jsonify({
        'ovarian_response': response,
        'ohss_risk': ohss,
        'response_confidence': round(resp_conf*100,1),
        'ohss_confidence': round(ohss_conf*100,1),
        'feature_importance': fi,
        'param_warnings': check_params(data)
    })

@app.route('/api/save_explanation/<patient_id>', methods=['POST'])
def save_exp(patient_id):
    if 'user' not in session: return jsonify({'error':'Unauthorized'}),401
    d = request.get_json()
    save_explanation(patient_id, d.get('explanation_text',''))
    return jsonify({'success':True})

@app.route('/api/get_explanations/<patient_id>')
def get_exps(patient_id):
    if 'user' not in session: return jsonify({'error':'Unauthorized'}),401
    exps = get_explanations(patient_id)
    return jsonify({'explanations': exps})

@app.route('/api/generate_explanation', methods=['POST'])
def generate_explanation():
    """Generate AI explanation server-side without needing Anthropic API from browser"""
    if 'user' not in session: return jsonify({'error':'Unauthorized'}),401
    d = request.get_json()
    patient = d.get('patient', {})
    question = d.get('question', 'Summarize this patient')
    
    # Generate rule-based clinical explanation
    name = patient.get('Patient_Name', 'Patient')
    age = patient.get('Age', 0)
    amh = float(patient.get('AMH') or 0)
    afc = int(patient.get('AFC') or 0)
    fsh = float(patient.get('Basal_FSH') or 0)
    lh = float(patient.get('Basal_LH') or 0)
    e2b = float(patient.get('Basal_E2') or 0)
    bmi = float(patient.get('BMI') or 0)
    stim = int(patient.get('Stimulation_Days') or 0)
    fs = float(patient.get('Dominant_Follicle_Size') or 0)
    e2t = float(patient.get('E2_Trigger') or 0)
    ooc = int(patient.get('Oocytes_Retrieved') or 0)
    ovarian_resp = patient.get('Ovarian_Response', 'Normal')
    ohss_risk = patient.get('OHSS_Risk', 0)

    q_lower = question.lower()
    
    # Determine response category
    if 'ohss' in q_lower or 'risk' in q_lower:
        reply = generate_ohss_explanation(name, afc, amh, e2t, ooc, ohss_risk)
    elif 'abnormal' in q_lower or 'parameter' in q_lower or 'param' in q_lower:
        reply = generate_abnormal_params(name, age, amh, afc, fsh, lh, e2b, bmi, stim, fs, e2t)
    elif 'recommend' in q_lower or 'treatment' in q_lower or 'protocol' in q_lower:
        reply = generate_recommendations(name, amh, afc, fsh, ovarian_resp, ohss_risk, bmi)
    elif 'summar' in q_lower or 'finding' in q_lower or 'overall' in q_lower:
        reply = generate_full_summary(name, age, bmi, amh, afc, fsh, lh, e2b, stim, fs, e2t, ooc, ovarian_resp, ohss_risk)
    elif 'response' in q_lower or 'ovarian' in q_lower or 'predict' in q_lower:
        reply = generate_response_explanation(name, amh, afc, fsh, ovarian_resp, stim, e2t, ooc)
    else:
        reply = generate_full_summary(name, age, bmi, amh, afc, fsh, lh, e2b, stim, fs, e2t, ooc, ovarian_resp, ohss_risk)
    
    return jsonify({'success': True, 'reply': reply})

def generate_ohss_explanation(name, afc, amh, e2t, ooc, ohss_risk):
    risk_level = "HIGH" if ohss_risk else "LOW"
    factors = []
    if afc > 20: factors.append(f"elevated AFC ({afc} follicles, normal: 5–30)")
    if amh > 4.5: factors.append(f"high AMH ({amh} ng/mL, indicates large ovarian reserve)")
    if e2t > 4000: factors.append(f"high E2 at trigger ({e2t} pg/mL, normal: 1000–4000)")
    if ooc > 15: factors.append(f"high oocyte yield ({ooc} retrieved)")
    
    text = f"**OHSS Risk Assessment for {name}**\n\n"
    text += f"📊 Risk Level: **{risk_level}**\n\n"
    
    if ohss_risk:
        text += "⚠️ This patient is flagged as HIGH risk for Ovarian Hyperstimulation Syndrome (OHSS).\n\n"
        if factors:
            text += "**Contributing risk factors:**\n"
            for f in factors: text += f"• {f}\n"
        text += "\n**Clinical Recommendations:**\n"
        text += "• Consider freeze-all embryo strategy\n"
        text += "• Use GnRH agonist trigger instead of hCG if possible\n"
        text += "• Withhold ET in fresh cycle; plan FET\n"
        text += "• Monitor closely with daily ultrasound post-retrieval\n"
        text += "• Consider cabergoline prophylaxis (0.5mg/day × 8 days)\n"
        text += "• Ensure adequate hydration and electrolyte monitoring\n"
    else:
        text += "✅ This patient shows LOW risk for OHSS based on current parameters.\n\n"
        text += "**Monitoring plan:**\n"
        text += "• Routine post-retrieval follow-up is sufficient\n"
        text += "• Standard luteal phase support\n"
        text += "• Fresh embryo transfer can be considered if indicated\n"
    return text

def generate_abnormal_params(name, age, amh, afc, fsh, lh, e2b, bmi, stim, fs, e2t):
    issues = []
    normal = []
    
    checks = [
        (age, 18, 45, 'Age', 'years'),
        (bmi, 18.5, 32, 'BMI', 'kg/m²'),
        (amh, 1.0, 6.0, 'AMH', 'ng/mL'),
        (afc, 5, 30, 'AFC', 'follicles'),
        (fsh, 2, 10, 'Basal FSH', 'IU/L'),
        (lh, 1, 10, 'Basal LH', 'IU/L'),
        (e2b, 20, 80, 'Basal E2', 'pg/mL'),
        (stim, 8, 14, 'Stimulation Days', 'days'),
        (fs, 17, 22, 'Dominant Follicle Size', 'mm'),
        (e2t, 1000, 4000, 'E2 at Trigger', 'pg/mL'),
    ]
    
    for val, lo, hi, label, unit in checks:
        if val and val > 0:
            if val < lo:
                issues.append(f"• ⬇ **{label}**: {val} {unit} — Below normal (ref: {lo}–{hi})")
            elif val > hi:
                issues.append(f"• ⬆ **{label}**: {val} {unit} — Above normal (ref: {lo}–{hi})")
            else:
                normal.append(f"• ✅ {label}: {val} {unit} — Normal")
    
    text = f"**Parameter Analysis for {name}**\n\n"
    if issues:
        text += f"**⚠️ {len(issues)} Abnormal Parameter(s):**\n"
        text += "\n".join(issues) + "\n\n"
    else:
        text += "✅ All entered parameters are within normal clinical reference ranges.\n\n"
    if normal:
        text += f"**Normal Parameters ({len(normal)}):**\n"
        text += "\n".join(normal)
    return text

def generate_recommendations(name, amh, afc, fsh, ovarian_resp, ohss_risk, bmi):
    text = f"**Treatment Recommendations for {name}**\n\n"
    
    # Protocol recommendation
    if ovarian_resp == 'Poor' or amh < 1.0 or fsh > 12:
        text += "**Stimulation Protocol:**\n"
        text += "• Consider antagonist protocol with high-dose FSH (300–450 IU/day)\n"
        text += "• Add LH supplementation (rLH or hMG)\n"
        text += "• Consider testosterone priming or DHEA supplementation\n"
        text += "• Dual trigger (GnRH agonist + low-dose hCG) may improve oocyte maturity\n\n"
    elif ovarian_resp == 'Hyper' or afc > 20 or amh > 4.5:
        text += "**Stimulation Protocol:**\n"
        text += "• Use low-dose stimulation (100–150 IU/day)\n"
        text += "• Antagonist protocol preferred to reduce OHSS risk\n"
        text += "• GnRH agonist trigger recommended\n"
        text += "• Elective freeze-all strategy strongly advised\n\n"
    else:
        text += "**Stimulation Protocol:**\n"
        text += "• Standard antagonist or long agonist protocol suitable\n"
        text += "• Starting dose: 150–225 IU/day, adjust per response\n"
        text += "• hCG trigger at lead follicle ≥18mm\n\n"
    
    if bmi > 30:
        text += "**Weight Management:**\n"
        text += "• BMI >30 increases anesthesia risk and reduces IVF success rates\n"
        text += "• Refer to dietitian; target 5–10% weight reduction pre-cycle\n\n"
    
    if ohss_risk:
        text += "**OHSS Prevention:**\n"
        text += "• Freeze all embryos; defer fresh transfer\n"
        text += "• Consider cabergoline 0.5mg/day for 8 days post-trigger\n"
        text += "• Daily monitoring post-retrieval\n\n"
    
    text += "**Monitoring Schedule:**\n"
    text += "• Baseline scan on day 2-3 of cycle\n"
    text += "• Stimulation monitoring every 2-3 days\n"
    text += "• Trigger when lead follicle ≥18mm with adequate E2\n"
    text += "• Retrieval 36 hours post-trigger\n"
    return text

def generate_response_explanation(name, amh, afc, fsh, ovarian_resp, stim, e2t, ooc):
    text = f"**Ovarian Response Explanation for {name}**\n\n"
    text += f"📊 Predicted Response: **{ovarian_resp}**\n\n"
    
    text += "**Key Factors Driving This Prediction:**\n"
    
    if amh > 0:
        amh_interp = "excellent reserve" if amh > 3 else "adequate reserve" if amh > 1.5 else "diminished reserve"
        text += f"• AMH ({amh} ng/mL): Indicates {amh_interp}\n"
    if afc > 0:
        afc_interp = "high pool" if afc > 20 else "good pool" if afc > 10 else "limited pool"
        text += f"• AFC ({afc}): Suggests {afc_interp} of antral follicles\n"
    if fsh > 0:
        fsh_interp = "low ovarian reserve signal" if fsh > 10 else "normal pituitary drive"
        text += f"• Basal FSH ({fsh} IU/L): {fsh_interp}\n"
    if stim > 0:
        text += f"• Stimulation Duration ({stim} days): {'Extended course' if stim > 12 else 'Standard duration'}\n"
    if e2t > 0:
        e2_interp = "robust follicular development" if e2t > 3000 else "moderate response" if e2t > 1500 else "limited response"
        text += f"• E2 at Trigger ({e2t} pg/mL): Suggests {e2_interp}\n"
    if ooc > 0:
        text += f"• Oocytes Retrieved ({ooc}): {'High yield' if ooc > 12 else 'Average yield' if ooc > 5 else 'Low yield'}\n"
    
    text += "\n**Clinical Interpretation:**\n"
    if ovarian_resp == 'Normal':
        text += "Patient shows expected response to stimulation. Standard monitoring and fresh transfer can be considered."
    elif ovarian_resp == 'Hyper':
        text += "Patient shows hyper-response. Risk of OHSS is elevated. Consider dose reduction and freeze-all strategy."
    else:
        text += "Patient shows poor response. Dose escalation, adjuvant therapy, or alternative protocols should be discussed with the clinical team."
    return text

def generate_full_summary(name, age, bmi, amh, afc, fsh, lh, e2b, stim, fs, e2t, ooc, ovarian_resp, ohss_risk):
    text = f"**Complete Clinical Summary — {name}**\n\n"
    text += f"**Demographics:** Age {age} yrs | BMI {bmi} kg/m²\n\n"
    text += f"**Ovarian Reserve:**\n"
    text += f"• AMH: {amh} ng/mL | AFC: {afc} follicles | Basal FSH: {fsh} IU/L\n"
    text += f"• Basal LH: {lh} IU/L | Basal E2: {e2b} pg/mL\n\n"
    text += f"**Stimulation Outcome:**\n"
    text += f"• Duration: {stim} days | Dominant Follicle: {fs} mm\n"
    text += f"• E2 at Trigger: {e2t} pg/mL | Oocytes Retrieved: {ooc}\n\n"
    text += f"**AI Predictions:**\n"
    text += f"• Ovarian Response: **{ovarian_resp}**\n"
    text += f"• OHSS Risk: **{'HIGH ⚠️' if ohss_risk else 'LOW ✅'}**\n\n"
    
    # Overall assessment
    flags = []
    if amh < 1.0: flags.append("diminished ovarian reserve (low AMH)")
    if amh > 5.0: flags.append("high ovarian reserve (possible PCOS pattern)")
    if fsh > 10: flags.append("elevated basal FSH")
    if bmi > 30: flags.append("BMI in obese range")
    if ohss_risk: flags.append("OHSS risk requiring preventive measures")
    
    if flags:
        text += "**Clinical Flags:**\n"
        for f in flags: text += f"• ⚠️ {f}\n"
        text += "\n"
    else:
        text += "✅ No major clinical flags identified.\n\n"
    
    text += "**Recommendation:** Review full clinical context with treating physician. "
    text += "This AI summary is a decision-support tool and must be validated by a qualified reproductive endocrinologist."
    return text

@app.route('/api/ocr', methods=['POST'])
def api_ocr():
    if 'user' not in session: return jsonify({'error':'Unauthorized'}),401
    if 'file' not in request.files: return jsonify({'error':'No file'}),400
    file = request.files['file']
    try:
        img = Image.open(file.stream).convert('RGB')
    except Exception as e:
        return jsonify({'error':f'Cannot open image: {str(e)}'}),400
    try:
        text = pytesseract.image_to_string(img)
    except Exception as e:
        return jsonify({'error':f'OCR failed: {str(e)}'}),500
    extracted = {}
    def try_patterns(patterns, cast):
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                try: return cast(m.group(1).strip())
                except: pass
        return None
    v = try_patterns([r'[Dd]ominant\s*[Ff]ollicle\s*[Ss]ize[\s:]+(\d+\.?\d*)',r'[Ff]ollicle[\s:]+(\d+\.?\d*)',r'(\d+\.?\d*)\s*mm'], float)
    if v: extracted['Dominant_Follicle_Size'] = v
    v = try_patterns([r'E2[\s:]+(\d+\.?\d*)',r'[Ee]stradiol[\s:]+(\d+\.?\d*)'], float)
    if v: extracted['E2_Trigger'] = v
    v = try_patterns([r'AFC[\s:]+(\d+)',r'[Aa]ntral[\s:]+(\d+)'], int)
    if v: extracted['AFC'] = v
    v = try_patterns([r'AMH[\s:]+(\d+\.?\d*)'], float)
    if v: extracted['AMH'] = v
    v = try_patterns([r'[Aa]ge[\s:]+(\d+)'], int)
    if v: extracted['Age'] = v
    v = try_patterns([r'[Pp]atient\s*[Nn]ame[\s:]+([A-Za-z][A-Za-z\s]{2,40})'], str)
    if v: extracted['Patient_Name'] = v.strip()
    v = try_patterns([r'FSH[\s:]+(\d+\.?\d*)'], float)
    if v: extracted['Basal_FSH'] = v
    v = try_patterns([r'LH[\s:]+(\d+\.?\d*)'], float)
    if v: extracted['Basal_LH'] = v
    v = try_patterns([r'BMI[\s:]+(\d+\.?\d*)'], float)
    if v: extracted['BMI'] = v
    fname = f"scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    img.save(os.path.join(UPLOAD_FOLDER, fname))
    buf = io.BytesIO()
    img.save(buf, 'JPEG', quality=85)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode()
    return jsonify({'success':True,'extracted':extracted,'raw_text':text[:600],'filename':fname,'image_b64':img_b64})

@app.route('/api/mock_hospital_data')
def mock_hospital_data():
    if 'user' not in session: return jsonify({'error':'Unauthorized'}),401
    import random
    data = {
        'Patient_Name': random.choice(['Priya Sharma','Ananya Patel','Deepa Nair','Kavitha Menon','Sunita Reddy']),
        'Age': random.randint(28,42),
        'BMI': round(random.uniform(19,30),1),
        'AMH': round(random.uniform(0.5,6.0),2),
        'AFC': random.randint(4,28),
        'Basal_FSH': round(random.uniform(3,15),2),
        'Basal_LH': round(random.uniform(1,12),2),
        'Basal_E2': round(random.uniform(20,80),1),
        'Stimulation_Days': random.randint(8,14),
        'Dominant_Follicle_Size': round(random.uniform(12,24),1),
        'E2_Trigger': round(random.uniform(800,5000),0),
        'Oocytes_Retrieved': random.randint(2,18),
    }
    warnings = check_params(data)
    return jsonify({'success':True,'data':data,'warnings':warnings})

@app.route('/api/report/<patient_id>')
def generate_report(patient_id):
    if 'user' not in session: return redirect(url_for('login'))
    patient = get_patient(patient_id)
    if not patient: return jsonify({'error':'Not found'}),404
    p = patient
    response, ohss, resp_conf, ohss_conf, _ = ml_predict(p)
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    story = []
    ts = ParagraphStyle('title',fontSize=20,textColor=colors.HexColor('#8B5A8C'),spaceAfter=6,alignment=1,fontName='Helvetica-Bold')
    story.append(Paragraph("IVF Clinical Decision Support Report", ts))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%d %B %Y, %I:%M %p')}",
                 ParagraphStyle('sub',fontSize=9,textColor=colors.grey,alignment=1)))
    story.append(Spacer(1,0.2*inch))
    info_data = [
        ['Patient ID',p['Patient_ID'],'Patient Name',p['Patient_Name']],
        ['Age',f"{int(p['Age'])} yrs",'BMI',f"{p['BMI']} kg/m²"],
        ['AMH',f"{p['AMH']} ng/mL",'AFC',str(p['AFC'])],
    ]
    t = Table(info_data, colWidths=[1.5*inch]*4)
    t.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E8D5F0')),
        ('GRID',(0,0),(-1,-1),0.5,colors.HexColor('#D0A8D8')),
        ('PADDING',(0,0),(-1,-1),6),
    ]))
    story.append(t)
    story.append(Spacer(1,0.2*inch))
    h2 = ParagraphStyle('h2',fontSize=13,textColor=colors.HexColor('#8B5A8C'),spaceAfter=4,fontName='Helvetica-Bold')
    story.append(Paragraph("AI Prediction Results",h2))
    res_data = [['Ovarian Response',response,'OHSS Risk','HIGH' if ohss else 'LOW']]
    t2 = Table(res_data, colWidths=[2*inch]*2)
    t2.setStyle(TableStyle([
        ('FONTNAME',(0,0),(0,-1),'Helvetica-Bold'),
        ('BACKGROUND',(0,0),(-1,-1),colors.HexColor('#FDF4FF')),
        ('GRID',(0,0),(-1,-1),0.5,colors.HexColor('#D0A8D8')),
        ('PADDING',(0,0),(-1,-1),6),
    ]))
    story.append(t2)
    story.append(Spacer(1,0.1*inch))
    story.append(Paragraph("This report is AI-generated and must be reviewed by a qualified clinician.",
                 ParagraphStyle('warn',fontSize=8,textColor=colors.grey)))
    doc.build(story)
    buf.seek(0)
    log_action('Generate Report', patient_id)
    return send_file(buf, as_attachment=True,
                     download_name=f"IVF_Report_{patient_id}.pdf", mimetype='application/pdf')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)