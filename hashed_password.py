import hashlib

def hash_password(password):
    # Generate SHA-256 hash of the password
    return hashlib.sha256(password.encode()).hexdigest()
print("Hashed admin password:", hash_password)