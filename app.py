from flask import Flask, flash, render_template, request, redirect, send_from_directory,session
import sqlite3
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "campushub_secret" 

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------- DATABASE ----------
def init_db():
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject TEXT,
            semester TEXT,
            filename TEXT,
            owner TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tutors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            subject TEXT,
            contact TEXT,
            owner TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            password TEXT
        )           
    """)

    conn.commit()
    conn.close()

    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()

    cursor.execute("""
CREATE TABLE IF NOT EXISTS errands (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    description TEXT,
    price INTEGER,
    contact TEXT,
    owner TEXT
)
""")
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS freelance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        description TEXT,
        price TEXT,
        contact TEXT,
        owner TEXT
    )
""")

    conn.commit()
    conn.close()

init_db()


# ---------- HOME PAGE ----------
@app.route("/")
def home():
    return render_template("index.html")


# ---------- NOTES PAGE ----------
@app.route("/notes", methods=["GET", "POST"])
def notes():
    if not session.get("user"):
        return redirect("/login")

    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()

    # Upload
    if request.method == "POST":

        subject = request.form["subject"]
        semester = request.form["semester"]
        file = request.files["file"]
        if not file or file.filename == "":
            return "No file selected", 400
        if not file.filename.lower().endswith('.pdf'):
            return "Only PDF files are allowed", 400

        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            owner = session.get("user")
            cursor.execute(
                "INSERT INTO notes (subject, semester, filename, owner) VALUES (?, ?, ?, ?)",
                (subject, semester, filename, owner)
            )
            
            conn.commit()
            return redirect("/notes")

    # Search
    search = request.args.get("search")
    if search:
        cursor.execute(
            "SELECT * FROM notes WHERE subject LIKE ?",
            ("%" + search + "%",)
        )
    else:
        cursor.execute("SELECT * FROM notes")

    notes = cursor.fetchall()
    notes_count = len(notes)
    conn.close()

    return render_template("notes.html", notes=notes, notes_count=notes_count)


# ---------- DELETE NOTE -----
@app.route("/delete/<int:id>")
def delete_note(id):

    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()

    # get filename + owner
    cursor.execute("SELECT filename, owner FROM notes WHERE id=?", (id,))
    note = cursor.fetchone()

    if not note:
        conn.close()
        return redirect("/notes")

    filename, owner = note

    # 🔥 OWNER CHECK
    if owner != session.get("user"):
        conn.close()
        return "Not allowed", 403

    # delete file
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(filepath):
        os.remove(filepath)

    # delete from DB
    cursor.execute("DELETE FROM notes WHERE id=?", (id,))
    conn.commit()
    conn.close()

    return redirect("/notes")


# ---------- SERVE UPLOADED FILES ----------
@app.route("/download/<filename>")
def download_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)
    


# ---------- OTHER PAGES ----------
@app.route("/tutoring", methods=["GET","POST"])
def tutoring():
    if not session.get("user"):
        return redirect("/login")

    conn = sqlite3.connect("database.db")
    cur = conn.cursor()
    
    if request.method == "POST":
        name = request.form["name"]
        subject = request.form["subject"]
        contact = request.form["contact"]

        # ✅ validation here ONLY
        if not contact.isdigit() or len(contact) != 10 or contact[0] not in ['6','7','8','9']:
            flash("Invalid contact number")
            conn.close()
            return redirect("/tutoring")

        if len(name.strip()) < 3 or not name.replace(" ", "").isalpha():
            flash("Enter valid name")
            conn.close()
            return redirect("/tutoring")

        if len(subject.strip()) < 2:
            flash("Enter valid subject")
            conn.close()
            return redirect("/tutoring")

        # ✅ THEN insert
        cur.execute(
            "INSERT INTO tutors(name,subject,contact,owner) VALUES (?,?,?,?)",
            (name, subject, contact, session.get("user"))
        )
        conn.commit()

    cur.execute("SELECT * FROM tutors")
    tutors = cur.fetchall()

    conn.close()

    return render_template("tutoring.html", tutors=tutors)

@app.route("/delete_tutor/<int:id>")
def delete_tutor(id):

    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()

    # get owner of tutor
    cursor.execute("SELECT owner FROM tutors WHERE id=?", (id,))
    tutor = cursor.fetchone()

    if not tutor:
        conn.close()
        return redirect("/tutoring")

    owner = tutor[0]

    # 🔥 IMPORTANT CHECK
    if owner != session.get("user"):
        conn.close()
        return "Not allowed", 403

    # delete if owner matches
    cursor.execute("DELETE FROM tutors WHERE id=?", (id,))
    conn.commit()
    conn.close()

    return redirect("/tutoring")
@app.route("/errands", methods=["GET", "POST"])
def errands():

    if "user" not in session:
        return redirect("/login")

    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()

    # ---------------- POST (ADD TASK) ----------------
    if request.method == "POST":

        title = request.form.get("title")
        description = request.form.get("description")
        price = request.form.get("price")
        contact = request.form.get("contact")
        owner = session.get("user")

        # ✅ validation
        import re
        if not re.match(r"^[6-9][0-9]{9}$", contact):
            return "Invalid phone number", 400

        if not title or not description:
            return "All fields required", 400

        cursor.execute("""
        INSERT INTO errands (title, description, price, contact, owner)
        VALUES (?, ?, ?, ?, ?)
        """, (title, description, price, contact, owner))

        conn.commit()

    # ---------------- GET (SHOW TASKS) ----------------
    cursor.execute("SELECT * FROM errands")
    tasks_list = cursor.fetchall()

    tasks = []
    for j in tasks_list:
        tasks.append({
            "id": j[0],
            "title": j[1],
            "description": j[2],
            "price": j[3],
            "contact": j[4],
            "owner": j[5]
        })

    conn.close()

    return render_template("errands.html", tasks=tasks)
@app.route("/delete_errand/<int:id>", methods=["POST"])
def delete_errand(id):

    if "user" not in session:
        return redirect("/login")

    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()

    cursor.execute("SELECT owner FROM errands WHERE id=?", (id,))
    task = cursor.fetchone()

    if not task:
        return "Not found", 404

    if task[0] != session.get("user"):
        return "Not allowed", 403

    cursor.execute("DELETE FROM errands WHERE id=?", (id,))
    conn.commit()
    conn.close()

    return redirect("/errands")


@app.route('/freelance', methods=['GET', 'POST'])
def freelance():
    if not session.get("user"):
        return redirect("/login")

    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()

    if request.method == 'POST':
        title = request.form['title']
        description = request.form['description']
        price = request.form['price']
        contact = request.form['contact']
        owner = session.get("user")

        cursor.execute("""
            INSERT INTO freelance (title, description, price, contact, owner)
            VALUES (?, ?, ?, ?, ?)
        """, (title, description, price, contact, owner))

        conn.commit()
        import re
        contact = request.form.get('contact')
        if not re.match(r'^[6-9]\d{9}$', contact):  
            return "Invalid contact number", 400

    cursor.execute("SELECT * FROM freelance")
    jobs_list = cursor.fetchall()

    jobs = []
    for j in jobs_list:
        jobs.append({
            "id": j[0],
            "title": j[1],
            "description": j[2],
            "price": j[3],
            "contact": j[4],
            "owner": j[5]
        })

    conn.close()
    
    
    

    return render_template("freelance.html", jobs=jobs)
@app.route('/delete_freelance/<int:id>', methods=['POST'])
def delete_freelance(id):
    if not session.get("user"):
        return redirect("/login")

    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()

    cursor.execute("SELECT owner FROM freelance WHERE id=?", (id,))
    job = cursor.fetchone()

    if not job:
        conn.close()
        return "Job not found", 404

    owner = job[0]

    if owner != session.get("user"):
        conn.close()
        return "Not allowed", 403

    cursor.execute("DELETE FROM freelance WHERE id=?", (id,))
    conn.commit()
    conn.close()

    return redirect('/freelance')
@app.route("/register", methods=["GET","POST"])
def register():

    if request.method == "POST":

        username = request.form["username"]
        password = request.form["password"]

        conn = sqlite3.connect("database.db")
        cursor = conn.cursor()

        cursor.execute(
        "INSERT INTO users (username,password) VALUES (?,?)",
        (username,password)
        )

        conn.commit()
        conn.close()

        return redirect("/login")

    return render_template("register.html")


@app.route("/login", methods=["GET","POST"])
def login():

    if request.method == "POST":

        username = request.form["username"]
        password = request.form["password"]

        conn = sqlite3.connect("database.db")
        cursor = conn.cursor()

        cursor.execute(
        "SELECT * FROM users WHERE username=? AND password=?",
        (username,password)
        )

        user = cursor.fetchone()

        conn.close()

        if user:
            session["user"]= username
            return redirect("/")
        else:
            return "Invalid login"

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect("/")



# ---------- RUN APP ----------
init_db()

if __name__ == "__main__":
    init_db()
    print("Database initialized and app is running...")
    port=int(os.environ.get("PORT", 500))
    app.run(host="0.0.0.0", port=500, debug=True)




