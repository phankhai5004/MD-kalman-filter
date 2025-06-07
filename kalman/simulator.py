import csv
import json
import requests
from datetime import datetime
import time


def read_csv_and_send_gps_data(csv_file_path, endpoint_url):
    """
    Read GPS data from CSV file and send each row as JSON to the specified endpoint
    """
    try:
        with open(csv_file_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)

            success_count = 0
            error_count = 0

            for row in reader:
                try:
                    # Convert data types appropriately
                    gps_data = {
                        "id": row["__id__"],
                        "accuracy": float(row["accuracy"]),
                        "latitude": float(row["latitude"]),
                        "longitude": float(row["longitude"]),
                        "altitude": float(row["altitude"]),
                        "course": float(row["course"]),
                        "fixtime": row["fixtime"],
                        "speed": float(row["speed"])
                    }

                    # Send POST request with JSON data
                    response = requests.post(
                        endpoint_url,
                        json=gps_data,
                        headers={'Content-Type': 'application/json'},
                        timeout=10
                    )

                    if response.status_code == 200:
                        print(f"✓ Successfully sent data for ID: {gps_data['id']}")
                        success_count += 1
                    else:
                        print(f"✗ Failed to send data for ID: {gps_data['id']} - Status: {response.status_code}")
                        error_count += 1

                    # Optional: Add small delay between requests to avoid overwhelming the server
                    time.sleep(0.1)

                except ValueError as e:
                    print(f"✗ Error parsing data for row: {row} - Error: {e}")
                    error_count += 1
                except requests.exceptions.RequestException as e:
                    print(f"✗ Network error sending data for ID: {row.get('__id__', 'unknown')} - Error: {e}")
                    error_count += 1

            print(f"\n=== Summary ===")
            print(f"Total successful sends: {success_count}")
            print(f"Total errors: {error_count}")

    except FileNotFoundError:
        print(f"Error: Could not find the file '{csv_file_path}'")
    except Exception as e:
        print(f"Error reading CSV file: {e}")


def send_single_record(csv_file_path, endpoint_url, record_id=None):
    """
    Send a single record (useful for testing)
    """
    try:
        with open(csv_file_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)

            for row in reader:
                if record_id is None or row["__id__"] == record_id:
                    gps_data = {
                        "id": row["__id__"],
                        "accuracy": float(row["accuracy"]),
                        "latitude": float(row["latitude"]),
                        "longitude": float(row["longitude"]),
                        "altitude": float(row["altitude"]),
                        "course": float(row["course"]),
                        "fixtime": row["fixtime"],
                        "speed": float(row["speed"])
                    }

                    print(f"Sending data: {json.dumps(gps_data, indent=2)}")

                    response = requests.post(
                        endpoint_url,
                        json=gps_data,
                        headers={'Content-Type': 'application/json'},
                        timeout=10
                    )

                    print(f"Response Status: {response.status_code}")
                    print(f"Response Body: {response.text}")
                    break
            else:
                if record_id:
                    print(f"Record with ID '{record_id}' not found")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # Configuration
    CSV_FILE_PATH = "newtrack_deduped.csv"
    ENDPOINT_URL = "http://localhost:8080/gps"

    print("GPS Data Sender")
    print("===============")
    print(f"CSV File: {CSV_FILE_PATH}")
    print(f"Endpoint: {ENDPOINT_URL}")
    print()

    # Choose mode
    mode = input(
        "Select mode:\n1. Send all records\n2. Send single record (for testing)\nEnter choice (1 or 2): ").strip()

    if mode == "1":
        print("\nSending all records...")
        read_csv_and_send_gps_data(CSV_FILE_PATH, ENDPOINT_URL)
    elif mode == "2":
        record_id = input("Enter record ID (or press Enter for first record): ").strip()
        if not record_id:
            record_id = None
        print(f"\nSending single record...")
        send_single_record(CSV_FILE_PATH, ENDPOINT_URL, record_id)
    else:
        print("Invalid choice. Running in send-all mode...")
        read_csv_and_send_gps_data(CSV_FILE_PATH, ENDPOINT_URL)