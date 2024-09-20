import subprocess
import time
import requests
import os

def start_server():
    # Change the working directory to the project root
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Start the server in a subprocess
    process = subprocess.Popen(
        ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"],
    )
    time.sleep(2)  # Give the server some time to start
    
    if process.poll() is not None:  # Check if the process has exited
        stdout, stderr = process.communicate()
        print(f"Error starting server: {stderr.decode()}")
        return
    
    # Wait for the server to start
    timer = 0
    server_started = False
    print("Server is starting up... Elapsed Time: 0 Seconds")
    while not server_started and timer < 600:
        time.sleep(1)  # Wait a second before checking again
        timer += 1
        # Check if the server is running
        try:
            response = requests.get("http://localhost:8000")
            if response.status_code == 200:
                server_started = True
                print("Server is up and running!")
                break  # Exit the loop once the server is running
        except requests.ConnectionError:
            if timer % 5 == 0:  # Print message every 5 seconds
                print(f"Waiting for server... {timer} Seconds")
    
    if not server_started:
        print("Error: timed out! Took more than 10 minutes :(")
        process.terminate()
        print("Server process terminated.")

if __name__ == "__main__":
    start_server()
