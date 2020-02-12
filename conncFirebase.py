import firebase_admin
from firebase_admin import credentials, firestore
import json

cred = credentials.Certificate('./supervisor-f2f29-firebase-adminsdk-l2twy-ae836f2735.json')


default_app = firebase_admin.initialize_app(cred)

db = firestore.client()

# when you want to add something create a similir set thing
doc_ref = db.collection(u'users').document(u'imageClassifier')

## Sets the data at this location to the given value.
#doc_ref.set({ u'name': "user", u'type': "Image name"})

## Returns the value, and optionally the ETag, at the current location of the database.
#for k in db.collection('users').stream():
#   print(k.to_dict())
#    print(k.id)

## Updates the specified child keys of this Reference to the provided values.
## doc_ref.update({ u'name': "user", u'type': "Cool Guy" })

## Creates a new child node.
# doc_ref.delete()

## Deletes this node from the database.
# doc_ref.push

print("Done")
