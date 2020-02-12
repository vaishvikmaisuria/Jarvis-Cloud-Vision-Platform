# JARVIS (Blockchain Based)


# Purpose:
    The purpose of this project is to build a scalable platform that would use existing hardware (provided by user) and Google Machine Learning APIs (such Google Vision, Google Voice, AutoML) to provide advanced applications, such as capturing Business meetings, or a food recipe that you used to cook a dish at home or essentially any idea that one can think of.


## Phase of Developement
    Using the google cloud API’s, build a detection from hazards such as fire, flooding,... (which ever is easy)
    Build a basic UI using react, firebase, firestore
    Connect it with that
    Work towards parametrizing the field of vision ( ie be able select a specific area of vision)
    Work towards, who is walking in and out of the area, create a log of that
    Then try to implement facial recognition such that if someone who is not in the recognized list enters the premises, security is notified
    Record everything on a private blockchain network

## Google Vision Tasks
    Detect Environment Status
    Fire
    Smoke
    Detecting who exists within parameter
    Known profiles
    Use facial detection, to determine if the person is known
    Unknown profiles
    Use profile,a
    Tracking objects (ie, be able to tell the supervisor to keep track of a specific object)
    Keeps tracks of till it can, then gives last seen location of the object


### Getting Started


You need to authenticate yourself to the firebase database so follow the link at: https://firebase.google.com/docs/admin/setup/


### Python imports you need to install (pip)
## Firebase
> pip3 install firebase_admin
## Twilio
> pip3 install twilio
#### Set the environment variable for sid from your twillio account (mac)
> export TWILIO_ACCOUNT_SID=GD8ef67043**************1942g5c267
> export TWILIO_AUTH_TOKEN=435***********************54325

## Google Cloud Setup
- pip install -m google-cloud (Google-Cloud model works with python2)
Follow the below links for setup
- https://cloud.google.com/vision/docs/quickstart-client-libraries
- https://cloud.google.com/docs/authentication/getting-started

