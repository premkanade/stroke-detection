from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from datetime import datetime


def create_pdf(
    filename,
    prediction,
    confidence,
    normal_prob,
    stroke_prob
):

    c = canvas.Canvas(filename, pagesize=letter)

    width, height = letter

    # Title
    c.setFont("Helvetica-Bold", 18)
    c.drawString(150, 770, "AI-Based Stroke Detection Report")

    # Date & Time
    c.setFont("Helvetica", 11)
    c.drawString(
        50,
        735,
        f"Generated On: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}"
    )

    # Line
    c.line(50, 720, 550, 720)

    # Prediction
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, 680, "Prediction")

    c.setFont("Helvetica", 12)
    c.drawString(70, 650, f"Result : {prediction}")
    c.drawString(70, 625, f"Confidence : {confidence:.2f}%")

    # Probabilities
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, 580, "Prediction Probabilities")

    c.setFont("Helvetica", 12)
    c.drawString(70, 550, f"Normal : {normal_prob:.2f}%")
    c.drawString(70, 525, f"Stroke : {stroke_prob:.2f}%")

    # Disclaimer
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, 460, "Disclaimer")

    c.setFont("Helvetica", 11)
    c.drawString(
        50,
        435,
        "This report is generated for educational and research purposes only."
    )

    c.drawString(
        50,
        415,
        "It should not replace professional medical diagnosis."
    )

    c.save()
    print("PDF Created Successfully")