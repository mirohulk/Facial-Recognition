# Facial-Recognition
This program uses advanced facial recognition technology to detect and identify faces in real-time from a live video feed. It compares detected faces against a database of known profiles and provides visual feedback directly on the video stream.

How It Works:

    1. Setup Known Profiles:
    	• Add images of known individuals to the Profiles folder.
    	• The program automatically encodes these images and associates them with the names provided in the script.
    
    2. Face Detection:
    	• Captures frames from the webcam.
    	• Detects face locations and scales them for faster processing.
    
    3. Recognition:
    	• Compares detected faces against the stored encodings.
    	• Matches are displayed with a green box and the person’s name.
    	• Unknown faces are marked with a red box and labeled as “Unrecognized.”
    
    4. Real-Time Feedback:
    	• Results are displayed directly on the video feed.
    	• Press q to quit the program.
  
Values to Update for Public Use:

    1. Known Profiles:
     
    	  • File Paths: Replace the image paths (image_paths) with publicly accessible directories or allow dynamic uploading of user images.
    	  • Names: Update the known_face_names list to match the corresponding names of individuals in the profile images.
       
           image_paths = ["path/to/user1.jpg", "path/to/user2.jpg"]
           known_face_names = ["User 1", "User 2"]
       
    3. Face Match Threshold:
     
    	  • Adjust the face_match_threshold for the desired accuracy:
    	  • Lower values (e.g., 0.4) are more strict, reducing false positives.
    	  • Higher values (e.g., 0.6) are more lenient, increasing recognition flexibility.
    
    4. Camera Source:
        
    	  •	Update the camera index in cv2.VideoCapture(0):
    	  •	0 for the default webcam.
    	  •	Other numbers (e.g., 1, 2) for external or multiple cameras.
      	
    5. Database Integration (Optional):
       
    	  •	Replace static lists (image_paths, known_face_names) with a dynamic database or API to load profiles.
    
    6. Output Labels:
    	  •	Customize the text displayed for recognized and unrecognized faces (e.g., change “Unrecognized” to “Guest” or “Unknown User”).
    
    7. Resolution and Display:
    	  •	Modify the resolution of the video feed for better clarity or performance based on hardware capabilities



Requirements:

    • Python installed on your computer.
    • A webcam for video input.

Required Libraries:

    • face_recognition
    • numpy
    • opencv-python

Usage:

	1. Add clear images of known individuals to the database.
	2. Run the program to start the facial recognition system.
	3. View the live video feed with recognition results.

Features:

	• Real-time face detection and recognition.
	• Simple setup for adding known profiles.
	• Visual feedback for recognized and unrecognized faces.

