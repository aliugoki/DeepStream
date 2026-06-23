import os
import psycopg2
import requests
import traceback
import time
import socket
import uuid
from datetime import datetime
from contextlib import contextmanager, closing
from psycopg2 import sql, extras

# --- CONFIGURATION (Now uses environment variables for security) ---
# It is highly recommended to set these values in your environment:
# e.g., export DB_USER="your_user"
# You can set default values here for local development if needed.
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "database": os.getenv("DB_NAME", "facial_recognition_db"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "admin@123"),
    "port": os.getenv("DB_PORT", "5432")
}
# --- Webhook Configuration ---
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "http://localhost:5001/api/add-entry")
WEBHOOK_HEADERS = {
    'Content-Type': 'application/json',
    'Authorization': f"Bearer {os.getenv('WEBHOOK_TOKEN', '6b17e44e-c22a-45a4-a301-c366e68c7c10')}"
}
WEBHOOK_TIMEOUT = 10
WEBHOOK_RETRIES = 3
INTERNET_CHECK_HOST = "8.8.8.8"
INTERNET_CHECK_PORT = 53
INTERNET_CHECK_TIMEOUT = 3

# --- COMPANY CONFIGURATION ---
# IMPORTANT: This variable should ideally be read from the environment
# or a configuration file in a production system.
COMPANY_ID = os.getenv("COMPANY_ID", "4c90df18-5fc0-47a9-a1ae-0f4e7c1cddf0")
# -----------------------------

# A global flag for simple, single-threaded database initialization.
_db_initialized = False

@contextmanager
def get_db_connection():
    """
    Provides a managed database connection using a context manager.
    The connection is automatically closed upon exiting the 'with' block.
    This also handles the one-time database initialization and migration.
    """
    global _db_initialized
    conn = None
    try:
        if not _db_initialized:
            print("\n--- Initializing/Migrating DB before first connection ---")
            # Ensure the database is ready before we proceed.
            if not _init_db_and_migrate():
                 print("Database initialization failed. Aborting connection attempt.")
                 return # Do not yield conn, the 'with' block will not have a connection.
            _db_initialized = True
            
        conn = psycopg2.connect(**DB_CONFIG)
        yield conn
    except psycopg2.OperationalError as e:
        print(f"ERROR: PostgreSQL connection failed: {e}")
        traceback.print_exc()
        # Do not yield conn, so the 'with' block won't have a connection.
    finally:
        if conn:
            conn.close()

def _init_db_and_migrate():
    """
    Handles the one-time database initialization and migration steps.
    This function is called by the `get_db_connection` context manager.
    It will now create the 'companies' and 'attendance1' tables.
    Returns True on success, False on failure.
    """
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        if not conn:
            return False

        with closing(conn.cursor()) as cursor:
            # Create companies table if it doesn't exist.
            # This is the parent table for user_data.
            # cursor.execute("""
            # CREATE TABLE IF NOT EXISTS companies (
            #     company_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            #     company_name VARCHAR(255) NOT NULL,
            #     admin_username VARCHAR(255) UNIQUE NOT NULL,
            #     admin_password_hash VARCHAR(255) NOT NULL,
            #     company_image_folder VARCHAR(255) NOT NULL,
            #     registration_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            #     api_key VARCHAR(255),
            #     status VARCHAR(50) DEFAULT 'active',
            #     rtsp_url TEXT,
            #     webrtc_url TEXT
            # )
            # """)

            # # Create user_data table if it doesn't exist.
            # # This is the parent table for attendance1.
            # cursor.execute("""
            # CREATE TABLE IF NOT EXISTS user_data (
            #     user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            #     company_id UUID NOT NULL,
            #     emp_id VARCHAR(255) NOT NULL,
            #     first_name VARCHAR(255) NOT NULL,
            #     last_name VARCHAR(255) NOT NULL,
            #     image_path TEXT,
            #     feature_path TEXT,
            #     registration_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            #     CONSTRAINT user_data_company_id_emp_id_key UNIQUE (company_id, emp_id),
            #     CONSTRAINT user_data_company_id_fkey FOREIGN KEY (company_id) REFERENCES companies(company_id) ON DELETE CASCADE
            # )
            # """)
            
            # --- CORRECTION: The FOREIGN KEY on emp_id must also include company_id to reference the unique constraint in user_data. ---
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
                 status TEXT,
                 FOREIGN KEY(emp_id, company_id) REFERENCES user_data(emp_id, company_id) ON DELETE CASCADE,
                 FOREIGN KEY(check_in_id) REFERENCES attendance1(id) ON DELETE CASCADE
             )
             """)

            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_attendance_emp_company ON attendance1(emp_id, company_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_attendance_date ON attendance1(attendance_date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_check_type ON attendance1(check_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_check_in_id ON attendance1(check_in_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_webhook_status ON attendance1(sent_to_webhook)")

            # Migrations to add new columns if they don't exist
            # Check for 'sent_to_webhook' column
            cursor.execute("""
                SELECT 1 FROM information_schema.columns 
                WHERE table_name='attendance1' AND column_name='sent_to_webhook'
            """)
            if not cursor.fetchone():
                print("Migrating database: Adding 'sent_to_webhook' column...")
                cursor.execute("""
                    ALTER TABLE attendance1
                    ADD COLUMN sent_to_webhook BOOLEAN DEFAULT FALSE
                """)

            # Check for 'webhook_response' column
            cursor.execute("""
                SELECT 1 FROM information_schema.columns 
                WHERE table_name='attendance1' AND column_name='webhook_response'
            """)
            if not cursor.fetchone():
                print("Migrating database: Adding 'webhook_response' column...")
                cursor.execute("""
                    ALTER TABLE attendance1
                    ADD COLUMN webhook_response TEXT
                """)
                
            # Insert a dummy company if not exists for the test
            company_uuid = uuid.UUID(COMPANY_ID)
            # FIX: Convert the UUID to a string for the SQL query
            cursor.execute("""
                INSERT INTO companies (company_id, company_name, admin_username, admin_password_hash, company_image_folder) 
                VALUES (%s, %s, %s, %s, %s) 
                ON CONFLICT (company_id) DO NOTHING
            """, (str(company_uuid), "Test Company", "admin", "hashed_password", f"company_images/{company_uuid}"))
            
            # Insert a dummy user if not exists for the test
            # FIX: Convert the UUID to a string for the SQL query
            cursor.execute("""
                INSERT INTO user_data (emp_id, company_id, first_name, last_name) 
                VALUES (%s, %s, %s, %s) 
                ON CONFLICT (company_id, emp_id) DO NOTHING
            """, (str(101), str(company_uuid), "John", "Doe"))
                
            conn.commit()
            print("Database initialization and migration completed successfully.")
            return True
    except Exception as e:
        print(f"ERROR: Error during database init/migration: {str(e)}")
        traceback.print_exc()
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()

def check_internet_connectivity(host=INTERNET_CHECK_HOST, port=INTERNET_CHECK_PORT, timeout=INTERNET_CHECK_TIMEOUT):
    """
    Checks for an active internet connection by trying to connect to a host.
    """
    try:
        socket.setdefaulttimeout(timeout)
        socket.create_connection((host, port), timeout)
        return True
    except socket.error as ex:
        print(f"Internet connectivity check failed: {ex}")
        return False
    
def send_to_webhook(attendance_data):
    """
    Sends attendance data to the webhook API with retry logic.
    Returns tuple: (success: bool, response: dict or str)
    """
    for attempt in range(WEBHOOK_RETRIES):
        try:
            response = requests.post(
                WEBHOOK_URL,
                headers=WEBHOOK_HEADERS,
                json=attendance_data,
                timeout=WEBHOOK_TIMEOUT
            )

            if response.status_code in [200, 201]:
                try:
                    return True, response.json()
                except ValueError:
                    return True, response.text
            else:
                print(f"Webhook Error (attempt {attempt + 1}): {response.status_code} - {response.text}")
                if attempt == WEBHOOK_RETRIES - 1:
                    return False, f"HTTP {response.status_code}: {response.text}"

        except requests.exceptions.RequestException as e:
            print(f"Webhook Connection Error (attempt {attempt + 1}): {str(e)}")
            if attempt == WEBHOOK_RETRIES - 1:
                return False, str(e)

        time.sleep(2 ** attempt)

    return False, "Max retries exceeded"

def mark_attendance_sent(attendance_id, webhook_response=None):
    """
    Updates an attendance record, marking it as sent and storing the webhook's response.
    """
    with get_db_connection() as conn:
        if not conn: return False
        try:
            with closing(conn.cursor()) as cursor:
                response_text = str(webhook_response) if webhook_response is not None else None
                if response_text and len(response_text) > 500:
                    response_text = response_text[:500] + "..."

                cursor.execute("""
                    UPDATE attendance1
                    SET sent_to_webhook = TRUE,
                        webhook_response = %s
                    WHERE id = %s
                """, (response_text, attendance_id))
                conn.commit()
                return True
        except Exception as e:
            print(f"ERROR: Error marking attendance as sent: {str(e)}")
            traceback.print_exc()
            conn.rollback()
            return False

def process_pending_webhooks():
    """
    Processes all attendance records not yet sent to the webhook.
    This now uses a single, more efficient database query with a JOIN.
    """
    with get_db_connection() as conn:
        if not conn: return

        try:
            # Use a dictionary cursor for easier access to column names
            with closing(conn.cursor(cursor_factory=psycopg2.extras.DictCursor)) as cursor:
                # The values are already the correct type, so no casting is needed in the query.
                cursor.execute("""
                    SELECT 
                        a.id, a.emp_id, a.company_id, a.first_name, a.last_name, 
                        a.attendance_date, a.attendance_time, a.check_type, 
                        a.camera_name, a.duration, a.check_in_id, 
                        u.image_path
                    FROM attendance1 AS a
                    JOIN user_data AS u ON a.emp_id = u.emp_id AND a.company_id = u.company_id
                    WHERE a.sent_to_webhook = FALSE
                    ORDER BY a.attendance_date, a.attendance_time
                """)

                records = cursor.fetchall()
                if not records:
                    print("No pending attendance records to send to webhook.")
                    return

                print(f"Found {len(records)} pending attendance records to send to webhook.")

                success_count = 0
                for record in records:
                    attendance_id = record['id']
                    payload = {
                        "id": attendance_id,
                        "emp_id": record['emp_id'],
                        "company_id": str(record['company_id']),
                        "first_name": record['first_name'],
                        "last_name": record['last_name'],
                        "attendance_date": record['attendance_date'].isoformat(),
                        "attendance_time": str(record['attendance_time']),
                        "check_type": record['check_type'],
                        "camera_name": record['camera_name'],
                        "duration": record['duration'],
                        "check_in_id": record['check_in_id'],
                        "timestamp": f"{record['attendance_date']} {record['attendance_time']}",
                        "system_source": "face_recognition_system",
                        "image_url": record['image_path']
                    }

                    print(f"Attempting to send pending attendance ID {attendance_id} to webhook...")
                    success, response = send_to_webhook(payload)
                    if success:
                        mark_attendance_sent(attendance_id, response)
                        success_count += 1
                        print(f"Successfully sent pending attendance ID {attendance_id} to webhook.")
                    else:
                        print(f"Failed to send pending attendance ID {attendance_id}. Will retry later. Error: {response}")

                print(f"Webhook processing complete. Success: {success_count}/{len(records)}.")

        except Exception as e:
            print(f"ERROR: Error processing pending webhooks: {str(e)}")
            traceback.print_exc()

def log_attendance(
    emp_id,
    company_id,
    first_name,
    last_name,
    attendance_date,
    attendance_time,
    check_type,
    image_url,
    camera_name=None,
    duration=None,
    check_in_id=None,    
):
    """
    Logs attendance and attempts to send data to the webhook immediately if
    internet connectivity is available.
    """
    with get_db_connection() as conn:
        if not conn: return False
        
        try:
            with closing(conn.cursor()) as cursor:
                attendance_id = None
                date_obj = datetime.strptime(attendance_date, "%d-%m-%Y").date()
                time_obj = datetime.strptime(attendance_time, "%H:%M:%S").time()

                # Convert emp_id and company_id to a string to match the column's data type
                emp_id_str = str(emp_id)
                company_id_str = str(company_id)

                if check_type == 'in':
                    # Check for duplicate check-ins
                    cursor.execute("""
                        SELECT COUNT(*) FROM attendance1
                        WHERE emp_id = %s AND company_id = %s AND attendance_date = %s AND check_type = 'in'
                    """, (emp_id_str, company_id_str, date_obj))
                    if cursor.fetchone()[0] > 0:
                        print(f"Employee {emp_id_str} in company {company_id_str} has already checked in today. Skipping.")
                        return False

                    cursor.execute("""
                        INSERT INTO attendance1 (
                            emp_id, company_id, first_name, last_name,
                            attendance_date, attendance_time, check_type, camera_name, image_url
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id
                    """, (
                        emp_id_str, company_id_str, first_name, last_name,
                        date_obj, time_obj, 'in', camera_name, image_url
                    ))

                    attendance_id = cursor.fetchone()[0]
                    print(f"Logged check-in for {first_name} {last_name} (Company {company_id_str}).")

                elif check_type == 'out':
                    # Find the latest open check-in
                    cursor.execute("""
                        SELECT id, attendance_time FROM attendance1
                        WHERE emp_id = %s AND company_id = %s AND attendance_date = %s AND check_type = 'in'
                        AND id NOT IN (SELECT check_in_id FROM attendance1 WHERE check_type = 'out' AND emp_id = %s AND company_id = %s AND attendance_date = %s)
                        ORDER BY attendance_time DESC
                        LIMIT 1
                    """, (emp_id_str, company_id_str, date_obj, emp_id_str, company_id_str, date_obj))
                    check_in_record = cursor.fetchone()

                    if not check_in_record:
                        print(f"No open check-in found for employee {emp_id_str} in company {company_id_str}. Cannot log check-out.")
                        return False
                    
                    check_in_id = check_in_record[0]
                    check_in_time = check_in_record[1]

                    # Check for duplicate check-outs
                    cursor.execute("""
                        SELECT COUNT(*) FROM attendance1
                        WHERE check_in_id = %s AND check_type = 'out' AND company_id = %s
                    """, (check_in_id, company_id_str))
                    if cursor.fetchone()[0] > 0:
                        print(f"Check-out already exists for check-in ID {check_in_id}. Skipping.")
                        return False
                    
                    # Calculate duration in minutes
                    duration_minutes = (datetime.combine(date_obj, time_obj) - datetime.combine(date_obj, check_in_time)).total_seconds() / 60
                    duration = round(duration_minutes, 2)

                    cursor.execute("""
                        INSERT INTO attendance1 (
                            emp_id, company_id, first_name, last_name,
                            attendance_date, attendance_time, check_type, 
                            camera_name, duration, check_in_id, image_url
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id
                    """, (
                        emp_id_str, company_id_str, first_name, last_name,
                        date_obj, time_obj, 'out', camera_name,
                        duration, check_in_id, image_url
                    ))

                    
                    attendance_id = cursor.fetchone()[0]
                    print(f"Logged check-out for {first_name} {last_name} (Company {company_id_str}), duration: {duration:.2f} mins.")

                conn.commit()
                
                # Retrieve the full record to prepare the webhook payload
                if attendance_id:
                    cursor.execute("""
                        SELECT id, emp_id, company_id, first_name, last_name, attendance_date, attendance_time,
                               check_type, camera_name, duration, check_in_id, sent_to_webhook, webhook_response
                        FROM attendance1 WHERE id = %s
                    """, (attendance_id,))
                    record_to_send = cursor.fetchone()

                    if record_to_send:
                        payload = prepare_webhook_payload(record_to_send)
                        if check_internet_connectivity():
                            print("Internet connected. Sending attendance to webhook immediately...")
                            success, response = send_to_webhook(payload)
                            if success:
                                mark_attendance_sent(attendance_id, response)
                                print(f"Successfully sent attendance ID {attendance_id} to webhook.")
                            else:
                                print(f"Failed to send attendance ID {attendance_id}. Will retry later. Error: {response}")
                        else:
                            print("No internet connection. Attendance stored locally. Will sync later.")
                return True

        except Exception as e:
            print(f"ERROR: Error logging attendance: {str(e)}")
            traceback.print_exc()
            conn.rollback()
            return False

def prepare_webhook_payload(attendance_record):
    """
    Prepares the attendance data for the webhook API.
    """
    emp_id = attendance_record[1]
    company_id = attendance_record[2]
    
    image_path = None
    with get_db_connection() as conn:
        if conn:
            with closing(conn.cursor()) as cursor:
                # FIX: Convert the UUID to a string for the SQL query
                cursor.execute("""
                    SELECT image_path 
                    FROM user_data 
                    WHERE emp_id = %s AND company_id = %s
                """, (emp_id, str(company_id)))
                result = cursor.fetchone()
                image_path = result[0] if result else None
    
    return {
        "id": attendance_record[0],
        "emp_id": str(emp_id),
        "company_id": str(company_id),
        "first_name": attendance_record[3],
        "last_name": attendance_record[4],
        "attendance_date": attendance_record[5].isoformat(),
        "attendance_time": str(attendance_record[6]),
        "check_type": attendance_record[7],
        "camera_name": attendance_record[8],
        "duration": attendance_record[9],
        "check_in_id": attendance_record[10],
        "timestamp": f"{attendance_record[5].isoformat()} {attendance_record[6]}",
        "system_source": "face_recognition_system",
        "image_url": image_path
    }
        
def check_attendance(emp_id, company_id, date=None):
    """
    Checks if an employee has an attendance record for a given date.
    """
    attendance_date = date or datetime.now().date()
    with get_db_connection() as conn:
        if not conn: return None
        try:
            with closing(conn.cursor()) as cursor:
                # Convert emp_id and company_id to string to match column type
                emp_id_str = str(emp_id)
                company_id_str = str(company_id)
                cursor.execute("""
                SELECT check_type, attendance_time 
                FROM attendance1
                WHERE emp_id = %s AND company_id = %s AND attendance_date = %s
                ORDER BY attendance_time
                """, (emp_id_str, company_id_str, attendance_date))
                return cursor.fetchall()
        except Exception as e:
            print(f"ERROR: Error checking attendance for emp {emp_id_str}: {str(e)}")
            traceback.print_exc()
            return None

def get_attendance_stats(company_id, date=None):
    """
    Gets attendance statistics for a given date, filtered by company.
    """
    attendance_date = date or datetime.now().date()
    with get_db_connection() as conn:
        if not conn: return None
        try:
            with closing(conn.cursor()) as cursor:
                # FIX: Convert the UUID to a string for the SQL query
                company_id_str = str(company_id)
                cursor.execute("SELECT COUNT(*) FROM user_data WHERE company_id = %s", (company_id_str,))
                total_employees = cursor.fetchone()[0]

                cursor.execute("""
                    SELECT COUNT(DISTINCT emp_id)
                    FROM attendance1
                    WHERE company_id = %s AND attendance_date = %s AND check_type = 'in'
                """, (company_id_str, attendance_date,))
                checked_in = cursor.fetchone()[0]

                cursor.execute("""
                    SELECT COUNT(DISTINCT emp_id)
                    FROM attendance1
                    WHERE company_id = %s AND attendance_date = %s AND check_type = 'out'
                """, (company_id_str, attendance_date,))
                checked_out = cursor.fetchone()[0]

                return {
                    'attendance_date': attendance_date,
                    'total_employees': total_employees,
                    'checked_in': checked_in,
                    'checked_out': checked_out,
                    'pending_checkouts': checked_in - checked_out
                }
        except Exception as e:
            print(f"ERROR: Error getting attendance stats for company {company_id_str}: {str(e)}")
            traceback.print_exc()
            return None

def get_user_info(emp_id, company_id):
    """
    Retrieves a user's information from the user_data table based on emp_id and company_id.
    Returns a tuple: (first_name, last_name, emp_id, image_path) — all None if not found or error.
    """
    with get_db_connection() as conn:
        if not conn:
            return (None, None, None, None)
        try:
            with closing(conn.cursor(cursor_factory=psycopg2.extras.DictCursor)) as cursor:
                emp_id_str = str(emp_id)
                company_id_str = str(company_id)

                cursor.execute("""
                    SELECT first_name, last_name, emp_id, image_path
                    FROM user_data
                    WHERE emp_id = %s AND company_id = %s
                """, (emp_id_str, company_id_str))

                user_record = cursor.fetchone()
                if user_record:
                    return (
                        user_record.get('first_name'),
                        user_record.get('last_name'),
                        user_record.get('emp_id'),
                        user_record.get('image_path')
                    )
                else:
                    return (None, None, None, None)
        except Exception as e:
            print(f"ERROR: Error getting user info for emp {emp_id}: {str(e)}")
            traceback.print_exc()
            return (None, None, None, None)

def main():
    """
    Main function to demonstrate the script's functionality.
    """
    current_date = datetime.now().date()
    current_time = datetime.now().time()

    print(f"\n--- Logging Attendance (Company: {COMPANY_ID}, Date: {current_date}) ---")

    # Simulate a check-in
    print("\nAttempting to log a check-in...")
    log_attendance(
        emp_id=101, # This will be converted to a string inside the function
        company_id=uuid.UUID(COMPANY_ID),
        first_name="John",
        last_name="Doe",
        attendance_date=current_date.strftime("%d-%m-%Y"),
        attendance_time=current_time.strftime("%H:%M:%S"),
        check_type='in',
        camera_name="Main Entrance"
    )

    # Simulate a check-out a few seconds later
    time.sleep(2)
    current_time_out = datetime.now().time()
    print(f"\nAttempting to log a check-out (Time: {current_time_out})...")
    log_attendance(
        emp_id=101, # This will be converted to a string inside the function
        company_id=uuid.UUID(COMPANY_ID),
        first_name="John",
        last_name="Doe",
        attendance_date=current_date.strftime("%d-%m-%Y"),
        attendance_time=current_time_out.strftime("%H:%M:%S"),
        check_type='out',
        camera_name="Main Exit"
    )

    print("\n--- Checking Attendance Records ---")
    records = check_attendance(101, uuid.UUID(COMPANY_ID), current_date)
    if records:
        for record_type, record_time in records:
            print(f"Employee 101 in Company {COMPANY_ID} on {current_date}: {record_type} at {record_time}")
    else:
        print(f"No attendance records found for employee 101 in company {COMPANY_ID} on {current_date}.")

    print("\n--- Getting Attendance Statistics ---")
    stats = get_attendance_stats(uuid.UUID(COMPANY_ID), current_date)
    if stats:
        print(f"Attendance Stats for Company {COMPANY_ID} on {stats['attendance_date']}:")
        print(f"Total Employees: {stats['total_employees']}")
        print(f"Checked In: {stats['checked_in']}")
        print(f"Checked Out: {stats['checked_out']}")
        print(f"Pending Checkouts: {stats['pending_checkouts']}")
    else:
        print("Could not retrieve attendance statistics.")

    print("\n--- Processing Pending Webhooks (if any) ---")
    # This call would be part of a separate scheduled task in a real system.
    process_pending_webhooks()

if __name__ == "__main__":
    main()
