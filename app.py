
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
    conn = sqlite3.connect("notes.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject TEXT,
            semester TEXT,
            filename TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tutors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            subject TEXT,
            contact TEXT
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

    conn = sqlite3.connect("notes.db")
    cursor = conn.cursor()

    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS errands (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            location TEXT,
            contact TEXT
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

    conn = sqlite3.connect("notes.db")
    cursor = conn.cursor()

    # Upload
    if request.method == "POST":

        subject = request.form["subject"]
        semester = request.form["semester"]
        file = request.files["file"]

        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))

            cursor.execute(
                "INSERT INTO notes (subject, semester, filename) VALUES (?, ?, ?)",
                (subject, semester, filename)
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


# ---------- DELETE NOTE ----------
@app.route("/delete/<int:id>")
def delete_note(id):

    conn = sqlite3.connect("notes.db")
    cursor = conn.cursor()

    cursor.execute("SELECT filename FROM notes WHERE id=?", (id,))
    file = cursor.fetchone()

    if file:
        filepath = os.path.join(UPLOAD_FOLDER, file[0])
        if os.path.exists(filepath):
            os.remove(filepath)

    cursor.execute("DELETE FROM notes WHERE id=?", (id,))
    conn.commit()
    conn.close()

    return redirect("/notes?success=1")


# ---------- SERVE UPLOADED FILES ----------
@app.route("/download/<filename>")
def download_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


# ---------- OTHER PAGES ----------
@app.route("/tutoring", methods=["GET","POST"])
def tutoring():

    conn = sqlite3.connect("notes.db")
    cur = conn.cursor()
    
    if request.method == "POST":
        name = request.form["name"]
        subject = request.form["subject"]
        contact = request.form["contact"]

        # ✅ validation here ONLY
        if not contact.isdigit() or len(contact) != 10 or contact[0] not in ['6','7','8','9']:
            flash("Invalid contact number")
            return redirect("/tutoring")

        if len(name.strip()) < 3 or not name.replace(" ", "").isalpha():
            flash("Enter valid name")
            return redirect("/tutoring")

        if len(subject.strip()) < 2:
            flash("Enter valid subject")
            return redirect("/tutoring")

        # ✅ THEN insert
        cur.execute(
            "INSERT INTO tutors(name,subject,contact) VALUES (?,?,?)",
            (name, subject, contact)
        )
        conn.commit()

    cur.execute("SELECT * FROM tutors")
    tutors = cur.fetchall()

    conn.close()

    return render_template("tutoring.html", tutors=tutors)

@app.route("/delete_tutor/<int:id>")
def delete_tutor(id):
    conn = sqlite3.connect("notes.db")
    cursor = conn.cursor()

    cursor.execute("DELETE FROM tutors WHERE id=?", (id,))
    conn.commit()
    conn.close()

    return redirect("/tutoring")

@app.route("/errands", methods=["GET","POST"])
def errands():

    conn = sqlite3.connect("notes.db")
    cursor = conn.cursor()

    if request.method == "POST":

        name = request.form["task"]
        location = request.form["location"]
        contact = request.form["contact"]

        cursor.execute(
        "INSERT INTO errands (name, location, contact) VALUES (?, ?, ?)",
        (name, location, contact)
        )

        conn.commit()

    cursor.execute("SELECT * FROM errands")
    errands = cursor.fetchall()

    conn.close()

    return render_template("errands.html", errands=errands)

@app.route("/freelance")
def freelance():
    return render_template("freelance.html")
@app.route("/register", methods=["GET","POST"])
def register():

    if request.method == "POST":

        username = request.form["username"]
        password = request.form["password"]

        conn = sqlite3.connect("notes.db")
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

        conn = sqlite3.connect("notes.db")
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
    app.run(debug=True)