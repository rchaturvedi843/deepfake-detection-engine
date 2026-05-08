from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import datetime

def generate_report(image_path, result, confidence, output_file):

    c = canvas.Canvas(output_file, pagesize=letter)

    width, height = letter

    c.setFont("Helvetica-Bold", 20)
    c.drawString(200, height-80, "Deepfake Detection Report")

    c.setFont("Helvetica", 12)

    c.drawString(50, height-150, f"Date: {datetime.datetime.now()}")

    c.drawString(50, height-180, f"Prediction Result: {result}")

    c.drawString(50, height-210, f"Confidence Score: {confidence:.2f}%")

    c.drawString(50, height-240, "Analysis Summary:")

    summary = """
This report was generated using a ResNet18 deep learning model trained
on a hybrid dataset of deepfake images and video frames.

The system analyzes facial artifacts, blending inconsistencies,
and texture anomalies to determine whether the media has been manipulated.
"""

    text = c.beginText(50, height-270)
    text.setFont("Helvetica", 11)

    for line in summary.split("\n"):
        text.textLine(line)

    c.drawText(text)

    c.drawImage(image_path, 50, 100, width=200, preserveAspectRatio=True)

    c.save()
