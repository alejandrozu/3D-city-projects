import http.server, functools, sys
CSP="default-src 'none'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src https://fonts.gstatic.com; img-src 'self' data: blob:; connect-src 'self'"
class H(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        if self.path.split('?')[0].endswith('.html'):
            self.send_header('Content-Security-Policy', CSP)
        super().end_headers()
    def log_message(self,*a): pass
http.server.ThreadingHTTPServer(('127.0.0.1',int(sys.argv[2])),functools.partial(H,directory=sys.argv[1])).serve_forever()
