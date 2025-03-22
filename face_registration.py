import os
from utils import capture_faces

def register_user(username, angles_required=5):
    os.makedirs('encodings', exist_ok=True)
    encoding_path = f'encodings/{username}_encodings.npy'

    if os.path.exists(encoding_path):
        print(f"[WARN] Username '{username}' already exists.")
        choice = input("Do you want to update existing face data? (y/n): ").strip().lower()

        if choice == 'y':
            print(f"[INFO] Updating face data for '{username}'.")
            capture_faces(username=username, angles_required=angles_required)
        else:
            new_username = input("Please enter a new username: ").strip()
            register_user(new_username, angles_required)
    else:
        capture_faces(username=username, angles_required=angles_required)

if __name__ == "__main__":
    username = input("Enter username for registration: ").strip()
    register_user(username=username)
