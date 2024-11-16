import hashlib
import os
import pickle
from pyexpat.errors import messages
import sqlite3
from datetime import datetime
from django import db
from django.shortcuts import render
from flask import Flask, flash, render_template, redirect, url_for, request, session, jsonify
from flask_login import LoginManager, UserMixin, login_user, login_required, current_user
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = os.environ.get(
    'SECRET_KEY', 'your_fallback_secret_key')  # Set secret key securely

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///your_database.db'
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'  # Redirect to login if not authenticated

# Define User model
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    role = db.Column(db.String(50), nullable=False)  # e.g., 'admin' or 'user'

# User loader function for Flask-Login
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# Function to load vectorizer and model


import numpy as np
import pickle

class LogisticRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None

    # Sigmoid function
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # Model prediction
    def predict(self, X):
        return self.sigmoid(np.dot(X, self.weights) + self.bias)

    # Train the model using gradient descent
    def train(self, X, y):
        m = X.shape[0]
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        # Gradient Descent
        for _ in range(self.iterations):
            # Linear model
            model = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(model)

            # Compute gradients
            dw = (1 / m) * np.dot(X.T, (predictions - y))
            db = (1 / m) * np.sum(predictions - y)

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    # Compute binary cross-entropy loss
    def compute_loss(self, X, y):
        m = X.shape[0]
        predictions = self.predict(X)
        loss = -(1 / m) * (np.dot(y.T, np.log(predictions)) + np.dot((1 - y).T, np.log(1 - predictions)))
        return loss

# Function to load the vectorizer and model
def load_vectorizer_and_model():
    try:
        vectorizer_path = 'feature_extraction.pkl'
        model_path = 'logistic_regression.pkl'

        # Check if the vectorizer file exists
        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(f"Could not find {vectorizer_path}")

        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)

        # Check if the model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Could not find {model_path}")

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # Ensure the vectorizer and model are loaded properly
        if vectorizer is None or model is None:
            raise ValueError("Failed to load vectorizer or model")

        return vectorizer, model

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None, None
    except Exception as e:
        print(f"Error loading model or vectorizer: {e}")
        return None, None


# Initialize the database
def init_db():
    conn = sqlite3.connect('your_database.db')
    cursor = conn.cursor()

    cursor.execute(''' 
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT NOT NULL
    )''')

    cursor.execute(''' 
    CREATE TABLE IF NOT EXISTS admin (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT NOT NULL
    )''')

    cursor.execute(''' 
    CREATE TABLE IF NOT EXISTS emails (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sender_id INTEGER,
        receiver_id INTEGER,
        subject TEXT,
        body TEXT,
        label TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(sender_id) REFERENCES users(id),
        FOREIGN KEY(receiver_id) REFERENCES users(id)
    )''')

    cursor.execute(''' 
    CREATE TABLE IF NOT EXISTS email_analysis_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email_text TEXT NOT NULL,
        prediction TEXT NOT NULL,
        date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')

    conn.commit()
    conn.close()

# Routes


@app.route('/')
def index():
    return render_template('index.html')


@app.route("/login", methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        try:
            with sqlite3.connect('your_database.db') as conn:
                cursor = conn.cursor()
                # Retrieve hashed password from database
                cursor.execute(
                    "SELECT id, username, password FROM users WHERE username = ?", (username,))
                user = cursor.fetchone()

                # Verify the user exists and password matches
                if user and check_password_hash(user[2], password):
                    session['username'] = username
                    session['user_id'] = user[0]
                    return redirect(url_for('user_dashboard'))
                else:
                    flash("Invalid credentials", "error")
        except sqlite3.Error as e:
            print(f"Database error: {e}")
            flash("An error occurred. Please try again.", "error")

    return render_template("login.html")

@app.route("/manage_users", methods=['GET', 'POST'])
def manage_users():
    success_message = ""
    error_message = ""

    if request.method == 'POST':
        action = request.form.get('action')
        username = request.form['username']
        
        # Establish a database connection
        conn = sqlite3.connect('your_database.db')
        cursor = conn.cursor()

        if action == 'add':
            password = request.form['password']
            hashed_password = hashlib.sha256(password.encode()).hexdigest()
            # Default role for new users
            role = 'user'  

            # Check if the user already exists
            cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
            if cursor.fetchone() is None:
                cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
                conn.commit()
                success_message = f"User '{username}' added successfully!"
            else:
                error_message = "Username already exists!"

        elif action == 'delete':
            # Delete the user with the given username
            cursor.execute("DELETE FROM users WHERE username = ?", (username,))
            if cursor.rowcount > 0:  # Check if any row was deleted
                conn.commit()
                success_message = f"User '{username}' deleted successfully!"
            else:
                error_message = "User not found!"

        conn.close()

    # Fetch users for displaying in the list
    conn = sqlite3.connect('your_database.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users")
    users = cursor.fetchall()
    conn.close()

    return render_template("manage_users.html", users=users, success_message=success_message, error_message=error_message)


@app.route("/admin_login", methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed_password = hashlib.sha256(password.encode()).hexdigest()

        try:
            conn = sqlite3.connect('your_database.db')
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, username FROM admin WHERE username = ? AND password = ?", (username, hashed_password))
            admin = cursor.fetchone()
            conn.close()

            if admin:
                session['username'] = username
                session['role'] = 'admin'
                session['admin_id'] = admin[0]
                return redirect(url_for('admin_dashboard'))
            else:
                flash("Invalid admin credentials!", "error")
                return render_template("admin_login.html")
        except sqlite3.Error as e:
            print(f"Database error: {e}")
            return "An error occurred with the database."
        except Exception as e:
            print(f"General error: {e}")
            return "An unexpected error occurred."
    return render_template("admin_login.html")





@app.route('/delete_account', methods=['GET', 'POST'])
@login_required
def delete_account():
    if request.method == 'POST':
        try:
            # Delete the user account
            db.session.delete(current_user)
            db.session.commit()
            
            flash('Your account has been deleted.', 'success')
            return redirect(url_for('home'))  # Redirect to the home page or login page

        except Exception as e:
            flash('An error occurred. Please try again.', 'danger')
            return redirect(url_for('user_dashboard'))  # Redirect back to dashboard if error

    return render_template('delete_account.html')


@app.route('/user_dashboard')
def user_dashboard():
    if 'username' not in session:
        flash("You need to be logged in to access the dashboard.", "warning")
        return redirect(url_for('login'))
    return render_template('user_dashboard.html', username=session['username'])




@app.route('/admin_dashboard')
def admin_dashboard():
    if 'username' not in session or session.get('role') != 'admin':
        return redirect(url_for('admin_login'))
    return render_template('admin_dashboard.html', username=session['username'])


# Load vectorizer and model globally
vectorizer, model = load_vectorizer_and_model()

# Function to classify email
def classify_email(mail):
    if vectorizer is None or model is None:
        flash("Model or vectorizer not loaded properly.", "error")
        return None

    try:
        # Predict the category of the email
        email_vec = vectorizer.transform([mail])
        # Debug: Show the vectorized input
        print("Vectorized Email:", email_vec.toarray())
        prediction = model.predict(email_vec)
        print("Prediction:", prediction)  # Debug: Show the prediction

        # Assign label based on prediction
        label = "ham" if prediction[0] == 1 else "spam"
        return label

    except Exception as e:
        print(f"Error during prediction: {e}")
        flash("Prediction failed. Please check the model and vectorizer.", "error")
        return None


@app.route('/compose', methods=['GET', 'POST'])
def compose():
    if 'user_id' not in session:
        flash("You need to be logged in to compose an email.", "warning")
        return redirect(url_for('login'))

    if request.method == 'POST':
        subject = request.form['subject']
        body = request.form['body']
        receiver_username = request.form['receiver']

        # Combine subject and body to classify the email
        email_text = subject + " " + body
       
        # yo dataset ma ham ra spam ko count hera ta...spam vanda dhera ham vayo vane ni sab ham huna sakxa
        # la disconnect gara ha ita huss kheii vayou vanya ma message garxu
        #  aba 6 pm paxi hai... huss
        
        label = classify_email(email_text)

        # Save the email to the database
        conn = sqlite3.connect('your_database.db')
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE username = ?",
                       (receiver_username,))
        receiver = cursor.fetchone()

        if receiver:
            receiver_id = receiver[0]
            sender_id = session['user_id']
            cursor.execute(""" 
                INSERT INTO emails (sender_id, receiver_id, subject, body, label, timestamp)
                VALUES (?, ?, ?, ?, ?, datetime('now'))
            """, (sender_id, receiver_id, subject, body, label))
            conn.commit()
            conn.close()

            flash(f"Email sent to {receiver_username} with subject: {subject}. Classified as {label}.", "success")
            # return redirect(url_for('inbox'))  # yo remove gara 
           
            
        else:
            flash(f"Receiver with username {receiver_username} not found.", "error")
            return render_template('compose.html')  # Ensure the response is returned

    # If GET request, just render the compose page
    return render_template('compose.html')


    

@app.route('/inbox')
def inbox():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']
    conn = sqlite3.connect('your_database.db')
    cursor = conn.cursor()
    cursor.execute(""" 
        SELECT emails.id, emails.subject, emails.body, emails.timestamp, emails.label, users.username
        FROM emails
        JOIN users ON emails.sender_id = users.id
        WHERE emails.receiver_id = ? 
        ORDER BY emails.timestamp DESC
    """, (user_id,))
    emails = cursor.fetchall()
    conn.close()

    return render_template('inbox.html', emails=emails)


@app.route("/register", methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed_password = generate_password_hash(password)

        try:
            with sqlite3.connect('your_database.db') as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
                conn.commit()
                flash("Registration successful! You can now log in.", "success")
                return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash("Username already taken, please choose another one.", "error")
        except sqlite3.Error as e:
            print(f"Database error: {e}")
            return "An error occurred with the database."

    return render_template("register.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out successfully.", "success")
    return redirect(url_for('login'))

# View email route


@app.route('/view_email/<int:email_id>')
def view_email(email_id):
    # Fetch the email from the database using the email_id
    conn = sqlite3.connect('your_database.db')
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, subject, body, timestamp, label, sender_id, receiver_id FROM emails WHERE id = ?", (email_id,))
    email = cursor.fetchone()  # email will now be a tuple
    conn.close()

    if email:
        # Render the view_email template with the email details
        # email is a tuple, you can now pass it to the template
        return render_template('view_email.html', email=email)
    else:
        flash("Email not found.", "error")
        return redirect(url_for('inbox'))


@app.route('/delete_email/<int:email_id>', methods=['POST'])
def delete_email(email_id):
    conn = sqlite3.connect('your_database.db')
    cursor = conn.cursor()

    try:
        cursor.execute("DELETE FROM emails WHERE id = ?", (email_id,))
        conn.commit()
        flash("Email deleted successfully.", "success")
    except Exception as e:
        flash(f"Error deleting email: {e}", "error")
    finally:
        conn.close()

    return redirect(url_for('inbox'))


if __name__ == '__main__':
    init_db()  # Initialize the database tables
    app.run(debug=True)
