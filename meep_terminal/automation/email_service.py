import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os

class EmailService:
    def __init__(self):
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", 587))
        self.sender_email = os.getenv("SENDER_EMAIL")
        self.password = os.getenv("SENDER_PASSWORD")

    def send_daily_picks_summary(self, recipient_email, picks):
        """Send a professional summary of the daily model output."""
        if not self.sender_email or not self.password:
            print("[WARN] Email credentials not configured. Skipping summary.")
            return

        msg = MIMEMultipart()
        msg['From'] = self.sender_email
        msg['To'] = recipient_email
        msg['Subject'] = f"🏀 MEEP | Daily Advisory - {len(picks)} Opportunities"

        # Construct HTML Body
        html = f"""
        <html>
        <body style="font-family: sans-serif; background-color: #f4f4f4; padding: 20px;">
            <div style="background-color: #1a1c24; color: white; padding: 20px; border-radius: 10px;">
                <h2>MEEP Terminal Report</h2>
                <p>Ensemble V4 has identified {len(picks)} qualifying opportunities for today's slate.</p>
                <table style="width: 100%; color: white; border-collapse: collapse;">
                    <tr style="border-bottom: 1px solid #3d3d5c;">
                        <th style="padding: 10px; text-align: left;">Player</th>
                        <th style="padding: 10px; text-align: left;">Prop</th>
                        <th style="padding: 10px; text-align: left;">Proj</th>
                        <th style="padding: 10px; text-align: left;">Heat Score</th>
                    </tr>
        """
        
        for pick in picks:
            html += f"""
                <tr style="border-bottom: 1px solid #2d2d44;">
                    <td style="padding: 10px;">{pick.get('player')}</td>
                    <td style="padding: 10px;">{pick.get('prop')}</td>
                    <td style="padding: 10px;">{pick.get('mu')}</td>
                    <td style="padding: 10px; color: #00ffcc;">{pick.get('heat')}</td>
                </tr>
            """

        html += """
                </table>
                <p style="margin-top: 20px; font-size: 0.8em; color: #888;">
                    This is an automated advisory. Please review the Terminal before execution.
                </p>
            </div>
        </body>
        </html>
        """
        
        msg.attach(MIMEText(html, 'html'))

        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.password)
                server.send_message(msg)
                print(f"[OK] Summary sent to {recipient_email}")
        except Exception as e:
            print(f"[ERR] Failed to send email: {e}")
