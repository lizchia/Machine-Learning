#!/usr/bin/env python3
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import os

PASSWORD_FILE = "password.txt"
PORT = 9000


def load_password():
    try:
        with open(PASSWORD_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return None


class AuthHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        
        # Redirect "/" to "/whisper_client.html"
        if parsed.path == "/":
            self.send_response(302)
            self.send_header("Location", "/whisper_client.html")
            self.end_headers()
            return

        # Endpoint used by the HTML to verify password
        if parsed.path == "/check_password":
            correct = load_password()
            if correct is None:
                self.send_error(500, "Password file missing")
                return

            qs = parse_qs(parsed.query)
            pw = qs.get("pw", [""])[0]

            if pw == correct:
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.end_headers()
                self.wfile.write(b"OK")
            else:
                self.send_response(403)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.end_headers()
                self.wfile.write(b"Forbidden")
            return

        # Do not serve the password file itself
        if parsed.path.endswith(PASSWORD_FILE):
            self.send_error(404)
            return

        # Everything else is served normally from current directory
        super().do_GET()


if __name__ == "__main__":
    # Serve files from the directory where this script lives
    web_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(web_root)

    httpd = HTTPServer(("0.0.0.0", PORT), AuthHandler)
    print(f"Serving on http://0.0.0.0:{PORT}")
    httpd.serve_forever()
