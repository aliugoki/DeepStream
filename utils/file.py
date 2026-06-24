import os
import sqlite3
import requests
import traceback
import time # Import time for sleep in retry logic
import socket # Import socket for internet connectivity check
from datetime import datetime
from contextlib import closing

# Database Configuration
DB_PATH = "/workspace/utils/iaa.db"

# Webhook Configuration
WEBHOOK_URL = "http://localhost:5001/api/add-entry"
# WEBHOOK_URL = WEBHOOK_URLS = [
#     "https://cloud.metaxperts.net:8443/erp/development/aiattendance/post/",
#     "http://localhost:5001/api/add-entry", # Example local server
# ]
WEBHOOK_HEADERS = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer MY$ecret'  # Replace with your auth token
}
WEBHOOK_TIMEOUT = 10  # seconds
WEBHOOK_RETRIES = 3  # number of retry attempts
# Host to check for internet connectivity (Google's public DNS)
INTERNET_CHECK_HOST = "8.8.8.8"
INTERNET_CHECK_PORT = 53
INTERNET_CHECK_TIMEOUT = 3 # seconds

def get_db_connection():
    """Establishes and returns a connection to the SQLite database."""
    return sqlite3.connect(DB_PATH)

def migrate_db():
    """
    Migrates the database to ensure the 'sent_to_webhook' and 'webhook_response'
    columns exist in the 'attendance1' table.
    """
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Check for 'sent_to_webhook' column
        cursor.execute("PRAGMA table_info(attendance1)")
        columns = [column[1] for column in cursor.fetchall()]

        migration_needed = False
        if 'sent_to_webhook' not in columns:
            print("Migrating database: Adding 'sent_to_webhook' column...")
            cursor.execute("""
                ALTER TABLE attendance1
                ADD COLUMN sent_to_webhook BOOLEAN DEFAULT 0
            """)
            migration_needed = True

        # Check for 'webhook_response' column
        if 'webhook_response' not in columns:
            print("Migrating database: Adding 'webhook_response' column...")
            cursor.execute("""
                ALTER TABLE attendance1
                ADD COLUMN webhook_response TEXT
            """)
            migration_needed = True

        if migration_needed:
            conn.commit()
            print("Database migration completed successfully.")
        else:
            print("Database is already up to date, no migration needed.")

    except Exception as e:
        print(f"Error during database migration: {str(e)}")
        traceback.print_exc()
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()

def init_db():
    """Initializes the database with required tables if they don't exist."""
    try:
        with closing(get_db_connection()) as conn:
            conn.execute("PRAGMA journal_mode=WAL")

            # Create user_data table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS user_data (
                emp_id INTEGER PRIMARY KEY,
                first_name TEXT NOT NULL,
                last_name TEXT NOT NULL,
                image_path TEXT,
                feature_path TEXT,
                registration_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)

            # Create attendance1 table
            # Added webhook_response column here for initial creation
            # Added ON DELETE CASCADE to foreign key constraints
            conn.execute("""
             CREATE TABLE IF NOT EXISTS attendance1 (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 emp_id INTEGER NOT NULL,
                 first_name TEXT NOT NULL,
                 last_name TEXT NOT NULL,
                 Attendance_date DATE NOT NULL,
                 Attendance_time TIMESTAMP NOT NULL,
                 check_type TEXT NOT NULL,
                 camera_name TEXT,
                 duration REAL,
                 check_in_id INTEGER,
                 sent_to_webhook BOOLEAN DEFAULT 0,  -- Renamed from sent_to_oracle
                 webhook_response TEXT,              -- New column for webhook response
                 FOREIGN KEY(emp_id) REFERENCES user_data(emp_id) ON DELETE CASCADE,
                 FOREIGN KEY(check_in_id) REFERENCES attendance1(id) ON DELETE CASCADE
             )
             """)

            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_attendance_emp ON attendance1(emp_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_Attendance_date ON attendance1(Attendance_date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_check_type ON attendance1(check_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_check_in_id ON attendance1(check_in_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_webhook_status ON attendance1(sent_to_webhook)")

            conn.commit()
    except Exception as e:
        print(f"Error initializing database: {str(e)}")
        traceback.print_exc()

def check_internet_connectivity(host=None, port=INTERNET_CHECK_PORT, timeout=INTERNET_CHECK_TIMEOUT):
    """
    Checks if there is an active internet connection by trying to connect
    to a well-known host.
    If 'host' is None, it defaults to the global INTERNET_CHECK_HOST.
    """
    check_host = host if host is not None else INTERNET_CHECK_HOST
    try:
        socket.setdefaulttimeout(timeout)
        socket.create_connection((check_host, port), timeout)
        return True
    except socket.error as ex:
        print(f"Internet connectivity check failed for {check_host}: {ex}")
        return False

def get_user_info(emp_id):
    """Retrieves first and last name for a given employee ID."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT first_name, last_name, emp_id FROM user_data WHERE emp_id = ?
        """, (emp_id,))
        result = cursor.fetchone()
        return result if result else (None, None, None)
    except Exception as e:
        print(f"Error getting user info for ID {emp_id}: {str(e)}")
        traceback.print_exc()
        return None, None
    finally:
        if conn:
            conn.close()

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
                except ValueError: # Handle cases where response is not JSON
                    return True, response.text
            else:
                print(f"Webhook Error (attempt {attempt + 1}): {response.status_code} - {response.text}")
                if attempt == WEBHOOK_RETRIES - 1:  # Last attempt
                    return False, f"HTTP {response.status_code}: {response.text}"

        except requests.exceptions.RequestException as e:
            print(f"Webhook Connection Error (attempt {attempt + 1}): {str(e)}")
            if attempt == WEBHOOK_RETRIES - 1:  # Last attempt
                return False, str(e)

        # Exponential backoff before retrying
        time.sleep(2 ** attempt)

    return False, "Max retries exceeded"

def mark_attendance_sent(attendance_id, webhook_response=None):
    """
    Updates an attendance record in the database, marking it as sent to the webhook
    and storing the webhook's response.
    """
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        # Truncate long responses to fit TEXT column, assuming 500 chars is enough
        response_text = str(webhook_response) if webhook_response is not None else None
        if response_text and len(response_text) > 500:
            response_text = response_text[:500] + "..."

        cursor.execute("""
            UPDATE attendance1
            SET sent_to_webhook = 1,
                webhook_response = ?
            WHERE id = ?
        """, (response_text, attendance_id))
        conn.commit()
        return True
    except Exception as e:
        print(f"Error marking attendance as sent: {str(e)}")
        traceback.print_exc()
        return False
    finally:
        if conn:
            conn.close()

def prepare_webhook_payload(attendance_record):
    """
    Prepares the attendance data into a dictionary suitable for the webhook API.
    Fetches image_path from user_data.
    """
    emp_id = attendance_record[1]
    
    # Fetch image_path for the user
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT image_path FROM user_data WHERE emp_id = ?", (emp_id,))
        result = cursor.fetchone()
        image_path = result[0] if result else None
    except Exception as e:
        print(f"Error fetching image_path for emp_id {emp_id}: {e}")
        image_path = None
    finally:
        if conn:
            conn.close()

    return {
        "id": attendance_record[0],
        "emp_id": emp_id,
        "first_name": attendance_record[2],
        "last_name": attendance_record[3],
        "attendance_date": attendance_record[4],
        "attendance_time": attendance_record[5],
        "check_type": attendance_record[6],
        "camera_name": attendance_record[7],
        "duration": attendance_record[8],
        "check_in_id": attendance_record[9],
        "timestamp": f"{attendance_record[4]} {attendance_record[5]}",
        "system_source": "face_recognition_system",
        "image_url": image_path  # ✅ Added
    }
        
def process_pending_webhooks():
    """
    Processes all attendance records not yet sent to the webhook.
    This function should be called when internet connectivity is detected or periodically.
    """
    #if not check_internet_connectivity():
    #    print("No internet connection. Skipping processing of pending webhooks.")
     #   return

    #conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Get all unsent records
        cursor.execute("""
            SELECT id, emp_id, first_name, last_name, Attendance_date, Attendance_time,
                   check_type, camera_name, duration, check_in_id, sent_to_webhook, webhook_response
            FROM attendance1
            WHERE sent_to_webhook = 0
            ORDER BY Attendance_date, Attendance_time
        """)

        records = cursor.fetchall()
        if not records:
            print("No pending attendance records to send to webhook.")
            return

        print(f"Found {len(records)} pending attendance records to send to webhook.")

        success_count = 0
        for record in records:
            attendance_id = record[0]
            payload = prepare_webhook_payload(record)

            print(f"Attempting to send pending attendance ID {attendance_id} to webhook...")
            success, response = send_to_webhook(payload)
            if success:
                mark_attendance_sent(attendance_id, response)
                success_count += 1
                print(f"Successfully sent pending attendance ID {attendance_id} to webhook.")
            else:
                print(f"Failed to send pending attendance ID {attendance_id} to webhook: {response}")

        print(f"Webhook processing complete. Success: {success_count}/{len(records)}.")

    except Exception as e:
        print(f"Error processing pending webhooks: {str(e)}")
        traceback.print_exc()
    finally:
        if conn:
            conn.close()


def log_attendance(
    emp_id,
    first_name,
    last_name,
    Attendance_date,  # DD-MM-YYYY format
    Attendance_time,  # HH:MM:SS format (24-hour)
    check_type,
    camera_name=None,
    duration=None,
    check_in_id=None
):
    """
    Logs complete attendance information with validation and handles webhook sending
    based on internet connectivity.
    """
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Convert to the desired format: "DD-MON-YYYY HH.MM.SS AM/PM"
        try:
            # Parse input date and time
            date_obj = datetime.strptime(Attendance_date, "%d-%m-%Y")
            time_obj = datetime.strptime(Attendance_time, "%H:%M:%S")

            # Format date as "DD-MON-YYYY" (e.g., "18-JUL-2025")
            formatted_date = date_obj.strftime("%d-%b-%Y").upper()

            # Format time as "HH.MM.SS AM/PM" (e.g., "10.14.28 AM")
            formatted_time = time_obj.strftime("%I.%M.%S %p")

            # Combine for database storage
            db_timestamp_display = f"{formatted_date} {formatted_time}"

        except ValueError as e:
            print(f"Error formatting timestamp: {e}")
            return False

        attendance_id = None # Initialize attendance_id

        if check_type == 'in':
            # Check for any check-in already today for this employee
            cursor.execute("""
                SELECT COUNT(*) FROM attendance1
                WHERE emp_id = ? AND Attendance_date = ? AND check_type = 'in'
            """, (emp_id, Attendance_date))
            if cursor.fetchone()[0] > 0:
                print(f"Employee {emp_id} has already checked in today. Skipping duplicate check-in.")
                return False

            # Insert check-in record. sent_to_webhook defaults to 0.
            cursor.execute("""
                INSERT INTO attendance1 (
                    emp_id, first_name, last_name,
                    Attendance_date, Attendance_time,
                    check_type, camera_name
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                emp_id, first_name, last_name,
                Attendance_date, db_timestamp_display,
                'in', camera_name
            ))
            attendance_id = cursor.lastrowid
            conn.commit()
            print(f"Logged check-in for {first_name} {last_name} at {db_timestamp_display}.")

        elif check_type == 'out':
            # If check_in_id is not provided, find the most recent open check-in
            if check_in_id is None:
                cursor.execute("""
                    SELECT id, Attendance_time FROM attendance1
                    WHERE emp_id = ? AND Attendance_date = ? AND check_type = 'in'
                    AND id NOT IN (SELECT check_in_id FROM attendance1 WHERE check_type = 'out' AND emp_id = ? AND Attendance_date = ?)
                    ORDER BY Attendance_time DESC
                    LIMIT 1
                """, (emp_id, Attendance_date, emp_id, Attendance_date))
                check_in_record = cursor.fetchone()

                if not check_in_record:
                    print(f"No open check-in found for employee {emp_id} on {Attendance_date}. Cannot log check-out.")
                    return False
                
                check_in_id = check_in_record[0] # Use the found check_in_id
                check_in_timestamp_display = check_in_record[1]
            else:
                # If check_in_id is provided, fetch its timestamp for duration calculation
                cursor.execute("""
                    SELECT Attendance_time FROM attendance1
                    WHERE id = ? AND check_type = 'in' AND emp_id = ?
                """, (check_in_id, emp_id))
                check_in_record = cursor.fetchone()
                
                if not check_in_record:
                    print(f"Error: Provided check_in_id {check_in_id} does not correspond to a valid check-in for emp_id {emp_id}.")
                    return False
                
                check_in_timestamp_display = check_in_record[0]

            # Check if a check-out already exists for this specific check_in_id
            cursor.execute("""
                SELECT COUNT(*) FROM attendance1
                WHERE check_in_id = ? AND check_type = 'out'
            """, (check_in_id,))
            if cursor.fetchone()[0] > 0:
                print(f"Check-out already exists for check-in ID {check_in_id}. Skipping duplicate check-out.")
                return False

            # Calculate duration
            try:
                # Parse the check-in display timestamp (e.g., "18-JUL-2025 10.14.28 AM")
                check_in_dt_calc = datetime.strptime(check_in_timestamp_display, "%d-%b-%Y %I.%M.%S %p")
                # Parse the check-out time (24-hour format)
                check_out_dt_calc = datetime.strptime(
                    f"{Attendance_date} {Attendance_time}",
                    "%d-%m-%Y %H:%M:%S"
                )
                duration = (check_out_dt_calc - check_in_dt_calc).total_seconds() / 60
                duration = round(duration, 2)
            except ValueError as e:
                print(f"Error calculating duration: {str(e)}")
                print(f"Check-in timestamp display: {check_in_timestamp_display}")
                print(f"Check-out time (input): {Attendance_time}")
                duration = 0.0

            # Insert check-out record. sent_to_webhook defaults to 0.
            cursor.execute("""
                INSERT INTO attendance1 (
                    emp_id, first_name, last_name,
                    Attendance_date, Attendance_time,
                    check_type, camera_name,
                    duration, check_in_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                emp_id, first_name, last_name,
                Attendance_date, db_timestamp_display,
                'out', camera_name,
                duration, check_in_id
            ))
            attendance_id = cursor.lastrowid
            conn.commit()
            print(f"Logged check-out for {first_name} {last_name}, duration: {duration:.2f} mins at {db_timestamp_display}.")

        # --- Webhook Sending Logic ---
        if attendance_id: # Only proceed if a record was successfully inserted
            # Retrieve the newly inserted record to send to webhook
            cursor.execute("""
                SELECT id, emp_id, first_name, last_name, Attendance_date, Attendance_time,
                       check_type, camera_name, duration, check_in_id, sent_to_webhook, webhook_response
                FROM attendance1 WHERE id = ?
            """, (attendance_id,))
            record_to_send = cursor.fetchone()

            if record_to_send:
                payload = prepare_webhook_payload(record_to_send)
                # Call check_internet_connectivity without explicitly passing a host
                # It will use the global INTERNET_CHECK_HOST by default
                if check_internet_connectivity():
                    print(f"Internet connected. Attempting to send attendance ID {attendance_id} to webhook immediately...")
                    success, response = send_to_webhook(payload)
                    if success:
                        mark_attendance_sent(attendance_id, response)
                        print(f"Successfully sent attendance ID {attendance_id} to webhook.")
                    else:
                        print(f"Failed to send attendance ID {attendance_id} to webhook. Will retry later. Error: {response}")
                else:
                    print(f"No internet connection. Attendance ID {attendance_id} stored locally. Will sync later.")
                    # sent_to_webhook is already 0 by default, no need to explicitly set

        return True

    except Exception as e:
        print(f"Error logging attendance: {str(e)}")
        traceback.print_exc()
        return False
    finally:
        if conn:
            conn.close()

def check_attendance(emp_id, date=None):
    """
    Checks if an employee has attendance record for a given date.
    Uses 'date' parameter, defaulting to today if not provided.
    """
    Attendance_date = date or datetime.now().strftime("%d-%m-%Y")
    try:
        with closing(get_db_connection()) as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT check_type, Attendance_time FROM attendance1
            WHERE emp_id = ? AND Attendance_date = ?
            ORDER BY Attendance_time
            """, (emp_id, Attendance_date))
            return cursor.fetchall()
    except Exception as e:
        print(f"Error checking attendance: {str(e)}")
        return None

def get_attendance_stats(date=None):
    """
    Gets attendance statistics for a given date.
    Uses 'date' parameter, defaulting to today if not provided.
    """
    Attendance_date = date or datetime.now().strftime("%d-%m-%Y")
    try:
        with closing(get_db_connection()) as conn:
            cursor = conn.cursor()

            # Total employees
            cursor.execute("SELECT COUNT(*) FROM user_data")
            total_employees = cursor.fetchone()[0]

            # Employees checked in today
            cursor.execute("""
                SELECT COUNT(DISTINCT emp_id)
                FROM attendance1
                WHERE Attendance_date = ? AND check_type = 'in'
            """, (Attendance_date,))
            checked_in = cursor.fetchone()[0]

            # Employees checked out today
            cursor.execute("""
                SELECT COUNT(DISTINCT emp_id)
                FROM attendance1
                WHERE Attendance_date = ? AND check_type = 'out'
            """, (Attendance_date,))
            checked_out = cursor.fetchone()[0]

            return {
                'Attendance_date': Attendance_date,
                'total_employees': total_employees,
                'checked_in': checked_in,
                'checked_out': checked_out,
                'pending_checkouts': checked_in - checked_out
            }
    except Exception as e:
        print(f"Error getting attendance stats: {str(e)}")
        return None

# Initialize database and run migrations when module loads
init_db()
migrate_db()

# Example Usage (for demonstration purposes, uncomment to test)
if __name__ == "__main__":
    print("\n--- Initializing/Migrating DB ---")
    init_db()
    migrate_db()

    # Add a dummy user if not exists
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT OR IGNORE INTO user_data (emp_id, first_name, last_name) VALUES (?, ?, ?)", (101, "John", "Doe"))
    conn.commit()
    conn.close()

    current_date = datetime.now().strftime("%d-%m-%Y")
    current_time = datetime.now().strftime("%H:%M:%S")

    print(f"\n--- Logging Attendance (Current Date: {current_date}, Current Time: {current_time}) ---")

    # Simulate a check-in
    print("\nAttempting to log a check-in...")
    log_attendance(
        emp_id=101,
        first_name="John",
        last_name="Doe",
        Attendance_date=current_date,
        Attendance_time=current_time,
        check_type='in',
        camera_name="Main Entrance"
    )

    # Simulate a check-out a few seconds later
    time.sleep(5) # Wait a bit to simulate time passing
    current_time_out = datetime.now().strftime("%H:%M:%S")
    print(f"\nAttempting to log a check-out (Current Time: {current_time_out})...")
    log_attendance(
        emp_id=101,
        first_name="John",
        last_name="Doe",
        Attendance_date=current_date,
        Attendance_time=current_time_out,
        check_type='out',
        camera_name="Main Exit"
    )

    print("\n--- Checking Attendance Records ---")
    records = check_attendance(101, current_date)
    if records:
        for record_type, record_time in records:
            print(f"Employee 101 on {current_date}: {record_type} at {record_time}")
    else:
        print(f"No attendance records found for employee 101 on {current_date}.")

    print("\n--- Getting Attendance Statistics ---")
    stats = get_attendance_stats(current_date)
    if stats:
        print(f"Attendance Stats for {stats['Attendance_date']}:")
        print(f"Total Employees: {stats['total_employees']}")
        print(f"Checked In: {stats['checked_in']}")
        print(f"Checked Out: {stats['checked_out']}")
        print(f"Pending Checkouts: {stats['pending_checkouts']}")
    else:
        print("Could not retrieve attendance statistics.")

    print("\n--- Processing Pending Webhooks (if any) ---")
    # This function would typically be called by a scheduler or on app startup
    # to sync any records that failed to send immediately.
    process_pending_webhooks()

    # Simulate no internet connection and logging attendance
    print("\n--- Simulating No Internet Connection (Conceptual) ---")
    print("In a real application, simulating no internet for `log_attendance` would involve mocking `check_internet_connectivity`.")
    print("For this example, we will proceed as if internet is available, and `log_attendance` will attempt to send.")
    print("Records that fail to send will be handled by `process_pending_webhooks` later.")

    # Log another check-in (this will now attempt to send if internet is truly available)
    time.sleep(2)
    current_time_after_sim_concept = datetime.now().strftime("%H:%M:%S")
    print(f"\nAttempting to log another check-in (Current Time: {current_time_after_sim_concept})...")
    log_attendance(
        emp_id=101,
        first_name="Jane",
        last_name="Doe",
        Attendance_date=current_date,
        Attendance_time=current_time_after_sim_concept,
        check_type='in',
        camera_name="Office Door"
    )

    print("\n--- Processing Pending Webhooks (Final Check) ---")
    process_pending_webhooks()
