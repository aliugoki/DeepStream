import os
import psycopg2
import requests
import traceback
import time
import socket
import uuid
from datetime import datetime
from contextlib import contextmanager, closing


last_processed_time = {}
# --- CONFIGURATION ---
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "database": os.getenv("DB_NAME", "facial_recognition_db"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "admin@123"),
    "port": os.getenv("DB_PORT", "5432")
}

WEBHOOK_URL = os.getenv("WEBHOOK_URL", "http://localhost:5001/api/add-entry")
WEBHOOK_HEADERS = {
    'Content-Type': 'application/json',
    'Authorization': f"Bearer {os.getenv('WEBHOOK_TOKEN', '9ff8ed8f-f8e7-4c1d-96da-b731da7b2fb9')}"
}

COMPANY_ID = os.getenv("COMPANY_ID", "267e6b1a-bc2e-4d18-b92b-7a37e3e4af7a")
WEBHOOK_RETRIES = 3
WEBHOOK_TIMEOUT = 10

_db_initialized = False

@contextmanager
def get_db_connection():
    global _db_initialized
    conn = None
    try:
        if not _db_initialized:
            _init_db_and_migrate()
            _db_initialized = True
        conn = psycopg2.connect(**DB_CONFIG)
        yield conn
    except Exception as e:
        print(f"DB Connection Error: {e}")
    finally:
        if conn: conn.close()

def _init_db_and_migrate():
    """Ensures tables exist without dummy data."""
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                 CREATE TABLE IF NOT EXISTS attendance1 (
                     id SERIAL PRIMARY KEY,
                     emp_id VARCHAR(255) NOT NULL,
                     company_id UUID NOT NULL,
                     first_name TEXT NOT NULL,
                     last_name TEXT NOT NULL,
                     attendance_date DATE NOT NULL,
                     attendance_time TIME NOT NULL,
                     check_type TEXT NOT NULL,
                     image_url TEXT,
                     camera_name TEXT,
                     duration REAL,
                     check_in_id INTEGER,
                     sent_to_webhook BOOLEAN DEFAULT FALSE,
                     webhook_response TEXT,
                     FOREIGN KEY(emp_id, company_id) REFERENCES user_data(emp_id, company_id) ON DELETE CASCADE
                 )
                 """)
                conn.commit()
    except Exception as e:
        print(f"Init DB Error: {e}")

# --- WEBHOOK LOGIC ---

def prepare_webhook_payload(record, image_path):
    """
    Standardizes payload using tuple indices.
    Order from SELECT: 0:id, 1:emp_id, 2:company_id, 3:first, 4:last, 5:date, 6:time, 7:type, 8:cam
    """
    return {
        "id": record[0],
        "emp_id": str(record[1]),
        "company_id": str(record[2]),
        "first_name": record[3],
        "last_name": record[4],
        "attendance_date": record[5].isoformat() if hasattr(record[5], 'isoformat') else str(record[5]),
        "attendance_time": record[6].isoformat() if hasattr(record[6], 'isoformat') else str(record[6]),
        "check_type": record[7],
        "camera_name": record[8],
        "image_url": image_path
    }

def send_to_webhook(payload):
    print(f"DEBUG: Attempting to send to {WEBHOOK_URL}...")
    for attempt in range(WEBHOOK_RETRIES):
        try:
            # Added verify=False in case of local SSL issues
            response = requests.post(
                WEBHOOK_URL, 
                headers=WEBHOOK_HEADERS, 
                json=payload, 
                timeout=WEBHOOK_TIMEOUT,
                verify=False 
            )
            print(f"DEBUG: Webhook Response Status: {response.status_code}")
            
            if response.status_code in [200, 201]:
                return True, response.text
            else:
                print(f"DEBUG: Webhook Error Body: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print(f"DEBUG: Connection Refused! Is the API running on {WEBHOOK_URL}?")
        except requests.exceptions.Timeout:
            print("DEBUG: Webhook timed out.")
        except Exception as e:
            print(f"DEBUG: Unexpected Webhook Error: {e}")
            
        time.sleep(1)
    return False, "Max retries reached"

def mark_attendance_sent(attendance_id, response_text):
    with get_db_connection() as conn:
        if not conn: return
        with conn.cursor() as cursor:
            cursor.execute("UPDATE attendance1 SET sent_to_webhook = True, webhook_response = %s WHERE id = %s", 
                           (str(response_text)[:500], attendance_id))
            conn.commit()

# --- CORE LOGIC ---

def get_user_info(emp_id, company_id):
    """Retrieves user details. Returns 4 values for probe.py compatibility."""
    with get_db_connection() as conn:
        if not conn: return (None, None, None, None)
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT first_name, last_name, emp_id, image_path 
                FROM user_data WHERE emp_id = %s AND company_id = %s
            """, (str(emp_id), str(company_id)))
            user = cursor.fetchone()
            # Index-based: 0=first, 1=last, 2=id, 3=path
            if user:
                return (user[0], user[1], user[2], user[3])
            return (None, None, None, None)

def log_attendance(emp_id, company_id, first_name=None, last_name=None, attendance_date=None, 
                   attendance_time=None, check_type='in', image_url=None, camera_name="Entrance"):
    
    # 1. THROTTLING: Prevent spamming the API (30-second cooldown)
    current_ts = time.time()
    if emp_id in last_processed_time and (current_ts - last_processed_time[emp_id] < 30):
        return False

    # 2. VALIDATION: Check if user exists
    user = get_user_info(emp_id, company_id)
    if not user[0]:
        print(f"EVENT IGNORED: ID {emp_id} not in user_data.")
        return False
    
    db_first, db_last, _, db_image = user
    now = datetime.now()
    date_obj = now.date()
    time_obj = now.time()

    with get_db_connection() as conn:
        if not conn: return False
        try:
            with conn.cursor() as cursor:
                # 3. AUTO-LOGIC: Determine if this is actually an 'in' or an 'out'
                # Look for the most recent check-in today that doesn't have a check-out yet
                cursor.execute("""
                    SELECT id FROM attendance1 
                    WHERE emp_id = %s AND company_id = %s AND attendance_date = %s 
                    AND check_type = 'in' AND id NOT IN (
                        SELECT check_in_id FROM attendance1 WHERE check_in_id IS NOT NULL
                    )
                    ORDER BY attendance_time DESC LIMIT 1
                """, (str(emp_id), str(company_id), date_obj))
                
                open_checkin = cursor.fetchone()
                
                # If they are already 'in', make this an 'out' event
                final_check_type = 'in'
                parent_id = None
                if open_checkin:
                    final_check_type = 'out'
                    parent_id = open_checkin[0]

                # 4. SAVE TO POSTGRES
                cursor.execute("""
                    INSERT INTO attendance1 (emp_id, company_id, first_name, last_name, attendance_date, 
                    attendance_time, check_type, camera_name, image_url, check_in_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id
                """, (str(emp_id), str(company_id), db_first, db_last, date_obj, time_obj, 
                      final_check_type, camera_name, db_image, parent_id))
                
                new_db_id = cursor.fetchone()[0]
                conn.commit()

                # 5. TRIGGER WEBHOOK
                last_processed_time[emp_id] = current_ts
                
                payload = {
                    "emp_id": str(emp_id),
                    "company_id": str(company_id),
                    "first_name": db_first,
                    "last_name": db_last,
                    "check_type": final_check_type,
                    "check_in_id": parent_id, # This fixes the 409 error for 'out' events
                    "attendance_date": date_obj.isoformat(),
                    "attendance_time": time_obj.isoformat(),
                    "image_url": db_image
                }
                
                success, resp = send_to_webhook(payload)
                if success:
                    mark_attendance_sent(new_db_id, resp)
                return True

        except Exception as e:
            print(f"Logging Error: {e}")
            return False