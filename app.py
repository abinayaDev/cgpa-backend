from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import pandas as pd
import os
import io
from supabase import create_client, Client, ClientOptions
from dotenv import load_dotenv
from prediction import predict_remaining_cgpa

from functools import wraps
import math
# Load env variables
load_dotenv()

app = Flask(__name__)
port = int(os.environ.get("PORT", 5000))
app.run(host="0.0.0.0", port=port)



# Security: Configure CORS - Allow all origins in development, restrict in production
frontend_url = os.getenv("FRONTEND_URL")

if os.getenv("FLASK_ENV") == "development":
    CORS(app, supports_credentials=True)
else:
    CORS(app, origins=[frontend_url], supports_credentials=True)


# Security: Rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"
)

# Security: File size limit (16MB)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

UPLOAD_FOLDER = "temp_uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"xlsx", "xls", "csv"}
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")

# Initialize Supabase with service role key (backend only)
if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Supabase credentials are missing. Check Render environment variables.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)



def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Security: JWT Token Verification Decorator
def verify_token(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({"message": "Missing or invalid authorization header"}), 401
        
        token = auth_header.replace('Bearer ', '')
        
        try:
            # Verify token with Supabase
            user_response = supabase.auth.get_user(token)
            request.user_id = user_response.user.id
            request.user_email = user_response.user.email
            request.token = token
        except Exception as e:
            return jsonify({"message": "Invalid or expired token"}), 401
            
        return f(*args, **kwargs)
    return decorated

def get_db():
    return create_client(
        SUPABASE_URL, 
        SUPABASE_KEY, 
        options=ClientOptions(headers={'Authorization': f'Bearer {request.token}'})
    )

# Security: Return anon key only (not service role key)
@app.route("/api/config", methods=["GET"])
def get_config():
    return jsonify({
        "url": SUPABASE_URL,
        "key": SUPABASE_ANON_KEY  # Public key only
    })

@app.route("/api/upload", methods=["POST"])
@limiter.limit("10 per minute")
@verify_token
def upload_file():
    # Get user_id from verified token
    user_id = request.user_id
    return _upload_and_parse_excel(user_id)

def _upload_and_parse_excel(user_id):
    file = request.files.get("file")
    if not file:
        return jsonify({"message": "No file uploaded"}), 400

    if not allowed_file(file.filename):
        return jsonify({"message": "Invalid file type"}), 400

    try:
        # Check file extension to determine parser
        if file.filename.endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
            
        # Normalize columns: lowercase and strip spaces
        df.columns = df.columns.str.lower().str.strip()
        
        # 1. Fix CGPA columns with spaces (e.g., "cgpa 1" -> "cgpa_1")
        df.columns = df.columns.str.replace(r'cgpa\s+(\d+)', r'cgpa_\1', regex=True)

        # Rename 'register number' or 'register_no' to 'roll_no' for consistency
        df.rename(columns={
            "register number": "roll_no", 
            "register_no": "roll_no",
            "roll number": "roll_no",
            "name": "student_name",
            "student name": "student_name"
        }, inplace=True)

        required_cols = {"student_name", "roll_no"}
        if not required_cols.issubset(df.columns):
            return jsonify({"message": f"Missing columns. Found: {list(df.columns)}. Required: Student Name, Register Number"}), 400

        results = []
        errors = []
        db_records = []

        for index, row in df.iterrows():
            excel_row = index + 2

            student_name = str(row["student_name"]).strip()
            roll_no = str(row["roll_no"]).strip()

            if not student_name or not roll_no or student_name == 'nan' or roll_no == 'nan':
                 if not student_name and not roll_no:
                    continue
                 errors.append(f"Row {excel_row}: Student name or Register Number missing")
                 continue

            completed = []
            missing = []

            for sem in range(1, 9):
                col = f"cgpa_{sem}"
                cgpa = row.get(col)

                if pd.isna(cgpa) or str(cgpa).strip() == "":
                    missing.append(sem)
                else:
                    try:
                        cgpa_val = float(cgpa)
                        if not (0 <= cgpa_val <= 10):
                             errors.append(f"Row {excel_row}: Invalid CGPA at semester {sem}. Must be 0-10.")
                             break
                        
                        completed.append({
                            "semester": sem,
                            "cgpa": cgpa_val
                        })
                        
                        # Prepare for DB Insert with USER_ID
                        db_records.append({
                            "user_id": user_id,
                            "roll_no": roll_no,
                            "student_name": student_name,
                            "semester": sem,
                            "cgpa": cgpa_val
                        })

                    except ValueError:
                         errors.append(f"Row {excel_row}: Invalid numeric CGPA at semester {sem}")
                         break

            if len(completed) < 1:
                errors.append(f"Row {excel_row}: At least one CGPA is required")
                continue

            results.append({
                "student_name": student_name,
                "roll_no": roll_no,
                "completed_semesters": completed,
                "missing_semesters": missing,
                "completed_count": len(completed),
                "remaining_count": len(missing)
            })

        if errors:
            return jsonify({
                "status": "error",
                "errors": errors
            }), 400

        duplicates_removed = 0
        inserted_count = 0

        # Perform Supabase Upsert
        if db_records:
            try:
                unique_records_list = list({ (r['roll_no'], r['semester']): r for r in db_records }.values())
                inserted_count = len(unique_records_list)
                duplicates_removed = len(db_records) - inserted_count

                db = get_db()
                data, count = db.table("academic_records").upsert(unique_records_list, on_conflict="user_id, roll_no, semester").execute()
            except Exception as e:
                return jsonify({"status": "error", "message": f"Database Insert Failed: {str(e)}"}), 500

        msg = f"Processed {inserted_count} unique records."
        if duplicates_removed > 0:
            msg += f" Merged {duplicates_removed} duplicate rows."

        return jsonify({
            "status": "success",
            "message": msg,
            "students_processed": len(results),
            "records_inserted": inserted_count,
            "duplicates_merged": duplicates_removed,
            "data_preview": results
        }), 200

    except Exception as e:
        return jsonify({"message": f"Error parsing file: {str(e)}"}), 500

@app.route("/api/predict/<roll_no>", methods=["GET"])
@verify_token
def predict_cgpa(roll_no):
    user_id = request.user_id
    email = request.user_email

    # Debug logging
    print(f"DEBUG: Authenticated user_id: {user_id}")
    print(f"DEBUG: Authenticated email: {email}")
    print(f"DEBUG: Looking for roll_no: {roll_no}")

    try:
        # Build Query
        db = get_db()
        query = db.table("academic_records").select("semester, cgpa, student_name, user_id").eq("roll_no", roll_no)
        
        # If NOT the special student admin, enforce data segregation
        if email != "studentit@gmail.com":
            query = query.eq("user_id", user_id)
            
        response = query.order("semester").execute()

        # Debug: Show what we found
        print(f"DEBUG: Found {len(response.data)} records")
        if response.data:
            print(f"DEBUG: First record user_id: {response.data[0].get('user_id')}")

        data = response.data

        if len(data) < 2:
            return jsonify({
                "message": "No academic records found for this Register Number. Please upload data first."
            }), 404

        student_name = data[0].get("student_name", "Unknown Student")

        completed = [
            {"semester": r["semester"], "cgpa": r["cgpa"]}
            for r in data
        ]

        predicted = predict_remaining_cgpa(completed)

        all_semesters = completed + [
            {"semester": p["semester"], "cgpa": p["predicted_cgpa"]}
            for p in predicted
        ]

        if all_semesters:
            final_cgpa = round(
                sum(s["cgpa"] for s in all_semesters) / len(all_semesters),
                2
            )
        else:
            final_cgpa = 0.0

        return jsonify({
            "roll_no": roll_no,
            "student_name": student_name,
            "completed_semesters": completed,
            "predicted_semesters": predicted,
            "final_cgpa": final_cgpa
        }), 200

    except Exception as e:
        return jsonify({"message": f"Prediction failed: {str(e)}"}), 500

@app.route("/api/export_all", methods=["GET"])
@verify_token
def export_all():
    user_id = request.user_id

    try:
        db = get_db()
        response = db.table("academic_records") \
            .select("roll_no, student_name, semester, cgpa") \
            .eq("user_id", user_id) \
            .order("roll_no, semester") \
            .execute()
        
        data = response.data
        if not data:
            return jsonify({"message": "No data found to export"}), 404

        students = {}
        for r in data:
            r_no = r['roll_no']
            if r_no not in students:
                students[r_no] = {
                    "roll_no": r_no,
                    "student_name": r.get('student_name', 'Unknown'),
                    "completed": []
                }
            students[r_no]["completed"].append({"semester": r['semester'], "cgpa": r['cgpa']})

        export_rows = []

        for r_no, student in students.items():
            completed = student['completed']
            completed.sort(key=lambda x: x['semester'])
            
            try:
                if len(completed) >= 2:
                    predicted = predict_remaining_cgpa(completed)
                else:
                    predicted = []
            except:
                predicted = []

            row = {
                "Register Number": student['roll_no'],
                "Student Name": student['student_name']
            }
            
            for c in completed:
                row[f"CGPA {c['semester']}"] = c['cgpa']
            
            for p in predicted:
                row[f"CGPA {p['semester']}"] = p['predicted_cgpa']
            
            export_rows.append(row)

        columns = ["Register Number", "Student Name"] + [f"CGPA {i}" for i in range(1, 9)]
        df_export = pd.DataFrame(export_rows)
        
        for col in columns:
            if col not in df_export.columns:
                df_export[col] = ""
        
        df_export = df_export[columns]

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_export.to_excel(writer, index=False, sheet_name='All Predictions')
        
        output.seek(0)
        
        return send_file(
            output, 
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name='all_students_prediction.xlsx'
        )

    except Exception as e:
        return jsonify({"message": f"Bulk Export failed: {str(e)}"}), 500

@app.route("/api/export_filtered", methods=["POST"])
@verify_token
def export_filtered():
    user_id = request.user_id
    data = request.json
    
    start_roll = data.get("start_roll", "").strip()
    end_roll = data.get("end_roll", "").strip()
    
    min_cgpa_str = str(data.get("min_cgpa", "")).strip()
    max_cgpa_str = str(data.get("max_cgpa", "")).strip()

    min_cgpa = float(min_cgpa_str) if min_cgpa_str else 0.0
    max_cgpa = float(max_cgpa_str) if max_cgpa_str else 10.0
    
    sort_order = data.get("sort_order", "none") # asc, desc, none

    try:
        db = get_db()
        response = db.table("academic_records") \
            .select("roll_no, student_name, semester, cgpa") \
            .eq("user_id", user_id) \
            .execute()
        
        records = response.data
        if not records:
            return jsonify({"message": "No data found"}), 404

        # Group by student
        students = {}
        for r in records:
            r_no = r['roll_no']
            if r_no not in students:
                students[r_no] = {
                    "roll_no": r_no,
                    "student_name": r.get('student_name', 'Unknown'),
                    "completed": []
                }
            students[r_no]["completed"].append({"semester": r['semester'], "cgpa": r['cgpa']})

        filtered_students = []

        for r_no, student in students.items():
            completed = student['completed']
            if not completed:
                continue

            # Calculate Average CGPA
            avg_cgpa = sum(c['cgpa'] for c in completed) / len(completed)

            # 1. Filter by Roll No Range (Alphabetical String Comparison)
            if start_roll and r_no < start_roll:
                continue
            if end_roll and r_no > end_roll:
                continue

            # 2. Filter by CGPA
            if avg_cgpa < min_cgpa or avg_cgpa > max_cgpa:
                continue

            # Prepare Export Row
            completed.sort(key=lambda x: x['semester'])
            
            try:
                if len(completed) >= 2:
                    predicted = predict_remaining_cgpa(completed)
                else:
                    predicted = []
            except:
                predicted = []

            row = {
                "Register Number": student['roll_no'],
                "Student Name": student['student_name'],
                "Average CGPA": round(avg_cgpa, 2)
            }
            
            for c in completed:
                row[f"CGPA {c['semester']}"] = c['cgpa']
            
            for p in predicted:
                row[f"CGPA {p['semester']}"] = p['predicted_cgpa']
            
            filtered_students.append(row)

        # 3. Sort
        if sort_order == "desc":
            filtered_students.sort(key=lambda x: x["Average CGPA"], reverse=True)
        elif sort_order == "asc":
            filtered_students.sort(key=lambda x: x["Average CGPA"])
        
        # Determine Columns
        columns = ["Register Number", "Student Name", "Average CGPA"] + [f"CGPA {i}" for i in range(1, 9)]
        df_export = pd.DataFrame(filtered_students)
        
        if df_export.empty:
             return jsonify({"message": "No students matched the filter criteria"}), 404

        for col in columns:
            if col not in df_export.columns:
                df_export[col] = ""
        
        df_export = df_export[columns]

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_export.to_excel(writer, index=False, sheet_name='Filtered Report')
        
        output.seek(0)
        
        return send_file(
            output, 
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name='filtered_report.xlsx'
        )

    except Exception as e:
        return jsonify({"message": f"Filter Export failed: {str(e)}"}), 500

@app.route("/api/simulate", methods=["POST"])
@verify_token
def simulate_prediction():
    try:
        data = request.json
        if not data or "completed_semesters" not in data:
            return jsonify({"message": "Invalid input. 'completed_semesters' list required."}), 400
        
        completed = data["completed_semesters"]
        
        for c in completed:
            if "semester" not in c or "cgpa" not in c:
                 return jsonify({"message": "Each item must have 'semester' and 'cgpa'"}), 400
            if not (0 <= float(c["cgpa"]) <= 10):
                 return jsonify({"message": "CGPA must be between 0 and 10"}), 400

        completed.sort(key=lambda x: x['semester'])

        if len(completed) < 2:
            return jsonify({"points_insufficient": True, "message": "Need at least 2 semesters to simulate prediction."}), 200

        predicted = predict_remaining_cgpa(completed)

        return jsonify({
            "completed_semesters": completed,
            "predicted_semesters": predicted
        }), 200

    except Exception as e:
        return jsonify({"message": f"Simulation failed: {str(e)}"}), 500

@app.route("/api/lock_semester", methods=["POST"])
@verify_token
def lock_semester():
    try:
        data = request.json
        user_id = request.user_id  # From verified token
        roll_no = data.get("roll_no")
        semester = data.get("semester")
        cgpa = data.get("cgpa")
        student_name = data.get("student_name", "Unknown")
        
        if not all([roll_no, semester, cgpa]):
            return jsonify({"message": "Missing required fields"}), 400
        
        if not (0 <= float(cgpa) <= 10):
            return jsonify({"message": "CGPA must be between 0 and 10"}), 400
        
        record = {
            "user_id": user_id,
            "roll_no": roll_no,
            "student_name": student_name,
            "semester": int(semester),
            "cgpa": float(cgpa)
        }
        
        db = get_db()
        response = db.table("academic_records").upsert(
            record, 
            on_conflict="user_id, roll_no, semester"
        ).execute()
        
        return jsonify({
            "message": "Semester locked successfully",
            "data": response.data
        }), 200
        
    except Exception as e:
        return jsonify({"message": f"Lock failed: {str(e)}"}), 500

# Security: Error handler
@app.errorhandler(Exception)
def handle_error(e):
    if os.getenv("FLASK_ENV") == "development":
        return jsonify({"message": str(e)}), 500
    return jsonify({"message": "Internal server error"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=(os.getenv("FLASK_ENV") == "development")
    )
