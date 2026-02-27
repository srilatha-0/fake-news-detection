import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Sender and receiver email details
sender_email = "srilathapmaula4@gmail.com"
receiver_email = "pamulavenky08@gmail.com"
password = "silh npna fehw rnyb"  # Your generated App Password

# Creating the email message
message = MIMEMultipart()
message["From"] = sender_email
message["To"] = receiver_email
message["Subject"] = "Automated Email Notification"

# Email body
body = "Hellooooooooooooo!"
message.attach(MIMEText(body, "plain"))

try:
    # Connecting to the Gmail SMTP server
    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()  # Secure the connection

    # Login using the app password
    server.login(sender_email, password)

    # Sending the email
    server.sendmail(sender_email, receiver_email, message.as_string())

    # Closing the connection
    server.quit()
    print("Email sent successfully!")

except Exception as e:
    print("Error:", e)
