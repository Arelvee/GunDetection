import subprocess

# Define the paths to your camera scripts
camera_scripts = ['python cam6.py',
                  'python app.py']


# Run each camera script concurrently
processes = [subprocess.Popen(script, shell=True) for script in camera_scripts]

# Wait for all processes to finish
for process in processes:
    process.wait()

print("All camera scripts have finished execution.")
