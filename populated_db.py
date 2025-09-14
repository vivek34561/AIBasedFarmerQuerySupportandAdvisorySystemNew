# populate_db.py

import sqlite3

def populate_users():
    """Inserts dummy user data into the users table."""
    
    # --- Define your dummy data here ---
    # Each tuple is: (phone_number, district, primary_crop)
    dummy_users = [
        (18,'+919151429036', 'Thrissur', 'Banana')
    ]

    try:
        # Connect to your existing database
        conn = sqlite3.connect('chatbot.db')
        cursor = conn.cursor()

        print("--- Populating 'users' table with dummy data ---")

        # Use executemany for efficient bulk insertion
        # The IGNORE clause prevents errors if a phone number already exists
        insert_query = "INSERT OR IGNORE INTO users (id , phone_number, district, primary_crop) VALUES (?, ?, ?, ?)"
        cursor.executemany(insert_query, dummy_users)

        # Commit the changes to the database
        conn.commit()

        # Print the number of rows added
        print(f"Successfully added {cursor.rowcount} new users.")

    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        # Always close the connection
        if conn:
            conn.close()
            print("--- Database connection closed ---")

if __name__ == "__main__":
    populate_users()