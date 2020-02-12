import os
from twilio.rest import Client


def send_msg(number, data):
    account_sid = os.environ["TWILIO_ACCOUNT_SID"]
    auth_token = os.environ["TWILIO_AUTH_TOKEN"]

    client = Client(account_sid, auth_token)

    # This my person phone number
    client.messages.create(
        to=number,
        from_="12563803381",
        body=data
        
    )

if __name__ == "__main__":
    send_msg("Vaishvik Got twilio working")
